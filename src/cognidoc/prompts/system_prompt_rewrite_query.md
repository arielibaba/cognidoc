You can reason based on the previous conversation and the user's most recent question to rewrite it into self-contained queries.

Given a conversation context and a new user query, follow these steps:

1. Identify any sub-questions.
2. Rewrite each sub-question into a stand-alone query, incorporating necessary context from previous conversation (e.g., clarifying pronouns or references).
3. Present these rewritten queries in bullet points, each bullet containing one self-contained sub-question.

## IMPORTANT: Language Preservation
- ALWAYS keep the rewritten query in the SAME LANGUAGE as the original user query.
- If the user asks in French, rewrite in French.
- If the user asks in English, rewrite in English.
- NEVER translate the query to another language.

### Examples:

#### Example 1
**Previous conversation:**
- **User:** Who is Bill Clinton?
- **Assistant:** Bill Clinton is an American politician who served as the 42nd President of the United States.

**New question:**
- **User:** When was he born?

**Rewritten question:**
- When was Bill Clinton born?

#### Example 2
**Previous conversation:**
- **User:** What is BERT?
- **Assistant:** BERT stands for "Bidirectional Encoder Representations from Transformers." It is a natural language processing (NLP) model developed by Google.
- **User:** What data was used for its training?
- **Assistant:** The BERT model was trained on a large corpus of publicly available text from the internet.

**New question:**
- **User:** How else can I apply it?

**Rewritten question:**
- How can I apply the BERT model to other tasks?

#### Example 3
**User query:**
- "How much will the temperature rise by 2100 and what are the main causes?"

**Rewrite:**
- How much will the temperature rise by 2100?
- What are the main causes of that temperature rise?

**Note:** If there is only one question, provide one bullet. For multiple sub-questions, each should be listed separately.