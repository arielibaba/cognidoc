"""
Unit tests for cognidoc.utils.chat_history module.

Tests cover:
- Conversation CRUD (create, list, rename, delete)
- Message add/get with source metadata
- Auto-title from first user message
- CASCADE delete of messages
- Export to JSON and Markdown formats
- Empty state handling
- Concurrent access via threading
"""

import json
import threading
import time
import uuid

import pytest

from cognidoc.utils.chat_history import ChatHistory


@pytest.fixture
def chat_history(tmp_path):
    """Create a ChatHistory instance backed by a temporary SQLite database."""
    db_path = str(tmp_path / "test_chat.db")
    return ChatHistory(db_path=db_path)


class TestCreateConversation:
    """Tests for conversation creation."""

    def test_create_conversation_returns_uuid(self, chat_history):
        """create_conversation should return a valid UUID string."""
        conv_id = chat_history.create_conversation()
        # Should not raise ValueError if it is a valid UUID
        parsed = uuid.UUID(conv_id)
        assert str(parsed) == conv_id

    def test_create_conversation_with_title(self, chat_history):
        """create_conversation with explicit title should store that title."""
        conv_id = chat_history.create_conversation(title="My Custom Title")
        conversations = chat_history.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["id"] == conv_id
        assert conversations[0]["title"] == "My Custom Title"

    def test_create_conversation_default_title(self, chat_history):
        """create_conversation without title should default to 'New Conversation'."""
        conv_id = chat_history.create_conversation()
        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert len(match) == 1
        assert match[0]["title"] == "New Conversation"


class TestListConversations:
    """Tests for listing conversations."""

    def test_empty_list_conversations(self, chat_history):
        """list_conversations on a fresh DB should return an empty list."""
        result = chat_history.list_conversations()
        assert result == []

    def test_list_conversations_ordered_by_date(self, chat_history):
        """Conversations should be listed most-recently-updated first."""
        id_a = chat_history.create_conversation(title="First")
        time.sleep(0.05)
        id_b = chat_history.create_conversation(title="Second")
        time.sleep(0.05)
        id_c = chat_history.create_conversation(title="Third")

        conversations = chat_history.list_conversations()
        assert len(conversations) == 3
        # Most recent first
        assert conversations[0]["id"] == id_c
        assert conversations[1]["id"] == id_b
        assert conversations[2]["id"] == id_a

    def test_list_conversations_respects_limit(self, chat_history):
        """list_conversations should respect the limit parameter."""
        for i in range(5):
            chat_history.create_conversation(title=f"Conv {i}")
            time.sleep(0.01)
        result = chat_history.list_conversations(limit=3)
        assert len(result) == 3


class TestMessages:
    """Tests for adding and retrieving messages."""

    def test_add_message_and_get_messages(self, chat_history):
        """Messages added to a conversation should be retrievable in order."""
        conv_id = chat_history.create_conversation()
        msg_id_1 = chat_history.add_message(conv_id, "user", "Hello!")
        time.sleep(0.01)
        msg_id_2 = chat_history.add_message(conv_id, "assistant", "Hi there!")

        assert isinstance(msg_id_1, int)
        assert isinstance(msg_id_2, int)
        assert msg_id_1 != msg_id_2

        messages = chat_history.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_add_message_with_sources(self, chat_history):
        """Messages with sources should store and retrieve them as parsed JSON."""
        conv_id = chat_history.create_conversation()
        sources = [
            {"file": "doc1.pdf", "page": 3, "score": 0.95},
            {"file": "doc2.pdf", "page": 1, "score": 0.87},
        ]
        chat_history.add_message(conv_id, "assistant", "Answer with sources", sources=sources)

        messages = chat_history.get_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["sources"] == sources
        assert isinstance(messages[0]["sources"], list)
        assert messages[0]["sources"][0]["file"] == "doc1.pdf"

    def test_add_message_without_sources_returns_none(self, chat_history):
        """Messages without sources should have sources as None."""
        conv_id = chat_history.create_conversation()
        chat_history.add_message(conv_id, "user", "No sources here")

        messages = chat_history.get_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["sources"] is None

    def test_get_messages_empty_conversation(self, chat_history):
        """get_messages on a conversation with no messages should return empty list."""
        conv_id = chat_history.create_conversation()
        messages = chat_history.get_messages(conv_id)
        assert messages == []

    def test_add_message_updates_conversation_timestamp(self, chat_history):
        """Adding a message should update the conversation's updated_at timestamp."""
        conv_id = chat_history.create_conversation()
        convs_before = chat_history.list_conversations()
        updated_before = convs_before[0]["updated_at"]

        time.sleep(0.05)
        chat_history.add_message(conv_id, "user", "Trigger update")

        convs_after = chat_history.list_conversations()
        updated_after = convs_after[0]["updated_at"]
        assert updated_after > updated_before


class TestAutoTitle:
    """Tests for automatic title generation from first user message."""

    def test_auto_title_from_first_user_message(self, chat_history):
        """First user message should auto-set the conversation title (up to 50 chars)."""
        conv_id = chat_history.create_conversation()
        chat_history.add_message(conv_id, "user", "What is quantum computing?")

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert match[0]["title"] == "What is quantum computing?"

    def test_auto_title_truncates_long_messages(self, chat_history):
        """Long first user messages should be truncated to 50 chars with ellipsis."""
        conv_id = chat_history.create_conversation()
        long_msg = "A" * 80
        chat_history.add_message(conv_id, "user", long_msg)

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert match[0]["title"] == "A" * 50 + "..."
        assert len(match[0]["title"]) == 53

    def test_auto_title_does_not_override_custom_title(self, chat_history):
        """Auto-title should NOT override a custom title set at creation."""
        conv_id = chat_history.create_conversation(title="My Custom Title")
        chat_history.add_message(conv_id, "user", "This should not become the title")

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert match[0]["title"] == "My Custom Title"

    def test_auto_title_only_on_first_user_message(self, chat_history):
        """Only the first user message should trigger auto-title (subsequent ones ignored)."""
        conv_id = chat_history.create_conversation()
        chat_history.add_message(conv_id, "user", "First question")
        chat_history.add_message(conv_id, "assistant", "First answer")
        chat_history.add_message(conv_id, "user", "Second question should not change title")

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert match[0]["title"] == "First question"

    def test_auto_title_skips_assistant_messages(self, chat_history):
        """An assistant message first should not trigger auto-title."""
        conv_id = chat_history.create_conversation()
        chat_history.add_message(conv_id, "assistant", "Welcome!")

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        # Title should remain "New Conversation" since no user message yet
        assert match[0]["title"] == "New Conversation"


class TestRenameConversation:
    """Tests for conversation renaming."""

    def test_rename_conversation(self, chat_history):
        """rename_conversation should update the conversation title."""
        conv_id = chat_history.create_conversation(title="Old Title")
        chat_history.rename_conversation(conv_id, "New Title")

        conversations = chat_history.list_conversations()
        match = [c for c in conversations if c["id"] == conv_id]
        assert match[0]["title"] == "New Title"

    def test_rename_updates_timestamp(self, chat_history):
        """rename_conversation should update the updated_at timestamp."""
        conv_id = chat_history.create_conversation()
        convs_before = chat_history.list_conversations()
        ts_before = convs_before[0]["updated_at"]

        time.sleep(0.05)
        chat_history.rename_conversation(conv_id, "Renamed")

        convs_after = chat_history.list_conversations()
        match = [c for c in convs_after if c["id"] == conv_id]
        assert match[0]["updated_at"] > ts_before


class TestDeleteConversation:
    """Tests for conversation deletion with CASCADE."""

    def test_delete_conversation_cascades_messages(self, chat_history):
        """Deleting a conversation should also delete all its messages (CASCADE)."""
        conv_id = chat_history.create_conversation()
        chat_history.add_message(conv_id, "user", "Message 1")
        chat_history.add_message(conv_id, "assistant", "Message 2")
        chat_history.add_message(conv_id, "user", "Message 3")

        # Verify messages exist before deletion
        assert len(chat_history.get_messages(conv_id)) == 3

        chat_history.delete_conversation(conv_id)

        # Conversation should be gone
        conversations = chat_history.list_conversations()
        assert all(c["id"] != conv_id for c in conversations)

        # Messages should also be gone (CASCADE)
        assert chat_history.get_messages(conv_id) == []

    def test_delete_does_not_affect_other_conversations(self, chat_history):
        """Deleting one conversation should not affect other conversations."""
        id_a = chat_history.create_conversation(title="Keep me")
        id_b = chat_history.create_conversation(title="Delete me")
        chat_history.add_message(id_a, "user", "Msg in A")
        chat_history.add_message(id_b, "user", "Msg in B")

        chat_history.delete_conversation(id_b)

        conversations = chat_history.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["id"] == id_a
        assert len(chat_history.get_messages(id_a)) == 1


class TestExportConversation:
    """Tests for conversation export in JSON and Markdown formats."""

    def test_export_json_format(self, chat_history):
        """export_conversation with JSON should return valid JSON with correct structure."""
        conv_id = chat_history.create_conversation()  # default "New Conversation"
        chat_history.add_message(conv_id, "user", "What is AI?")
        chat_history.add_message(conv_id, "assistant", "AI is artificial intelligence.")

        result = chat_history.export_conversation(conv_id, fmt="json")
        data = json.loads(result)

        assert data["id"] == conv_id
        assert data["title"] == "What is AI?"  # auto-titled from first user msg
        assert "created_at" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "What is AI?"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "AI is artificial intelligence."

    def test_export_markdown_format(self, chat_history):
        """export_conversation with Markdown should return properly formatted text."""
        conv_id = chat_history.create_conversation(title="Markdown Test")
        chat_history.add_message(conv_id, "user", "Tell me about Python.")
        chat_history.add_message(conv_id, "assistant", "Python is a programming language.")

        result = chat_history.export_conversation(conv_id, fmt="markdown")

        assert result.startswith("# Markdown Test\n")
        assert "**User**:\nTell me about Python." in result
        assert "**Assistant**:\nPython is a programming language." in result

    def test_export_nonexistent_conversation_returns_empty(self, chat_history):
        """Exporting a non-existent conversation should return an empty string."""
        result = chat_history.export_conversation("nonexistent-id", fmt="json")
        assert result == ""

    def test_export_json_includes_sources(self, chat_history):
        """Exported JSON should include parsed sources on messages."""
        conv_id = chat_history.create_conversation(title="Sources Export")
        sources = [{"file": "report.pdf", "page": 5}]
        chat_history.add_message(conv_id, "assistant", "Here is the info.", sources=sources)

        result = chat_history.export_conversation(conv_id, fmt="json")
        data = json.loads(result)
        assert data["messages"][0]["sources"] == sources


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access to the SQLite database."""

    def test_concurrent_access(self, tmp_path):
        """Multiple threads should be able to read/write without corruption."""
        db_path = str(tmp_path / "concurrent_chat.db")
        history = ChatHistory(db_path=db_path)

        num_threads = 8
        messages_per_thread = 10
        errors = []
        conv_ids = []

        # Each thread creates a conversation and adds messages
        def worker(thread_idx):
            try:
                conv_id = history.create_conversation(title=f"Thread {thread_idx}")
                conv_ids.append(conv_id)
                for i in range(messages_per_thread):
                    history.add_message(conv_id, "user", f"Thread {thread_idx} msg {i}")
            except Exception as e:
                errors.append((thread_idx, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No errors should have occurred
        assert errors == [], f"Errors during concurrent access: {errors}"

        # All conversations should exist
        conversations = history.list_conversations(limit=100)
        assert len(conversations) == num_threads

        # Each conversation should have the correct number of messages
        total_messages = 0
        for conv in conversations:
            msgs = history.get_messages(conv["id"])
            assert len(msgs) == messages_per_thread
            total_messages += len(msgs)

        assert total_messages == num_threads * messages_per_thread
