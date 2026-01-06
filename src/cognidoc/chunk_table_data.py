from pathlib import Path
from typing import Dict, List


import ollama

from .helpers import (
    ask_LLM_with_JSON,
    recover_json,
    get_token_count
)



def chunk_markdown_table_with_overlap(md_table, cols=None, n_tokens=512, overlap=128):
    """
    Splits a markdown table into chunks with overlapping tokens.
    Each returned chunk already has the header prepended.

    Parameters:
        - md_table (str): The markdown table as a string.
        - cols (list of str], optional): List of column headers. If provided, overrides the header in md_table.
        - n_tokens (int): Maximum number of tokens per chunk (including header tokens).
        - overlap (int): Number of tokens to overlap between adjacent chunks (data‐rows only).

    Returns:
        - chunks (list of str): Each element is a string consisting of
                                (header + data‐rows with overlaps).
        - header (str): The header block (header row + separator row).
    """
    # Split into non‐empty, stripped lines
    mds = [line.rstrip() for line in md_table.strip().split("\n") if line.strip()]
    if not mds:
        return [], ""

    # Build header based on cols override or the first two lines of md_table
    if cols is not None:
        # Use the provided cols to generate a new header
        header_row = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        header = header_row + "\n" + separator + "\n"
        # Skip the first two lines of mds (the original header + separator)
        data_start_idx = 2
    else:
        # Use the existing first line as header; check if second line is the separator
        header = mds[0] + "\n"
        if len(mds) > 1 and all(ch in "-:|" for ch in mds[1].replace(" ", "")):
            header += mds[1] + "\n"
            data_start_idx = 2
        else:
            data_start_idx = 1

    # Count how many tokens the header takes (so each chunk can include that cost)
    header_token_count = get_token_count(header)

    chunks = []                  # Will hold final strings: (header + data‐rows)
    current_chunk_data = []      # Collects just the data‐rows (each ending in "\n")
    current_token_count = header_token_count

    # Iterate over each data row from the markdown table
    for raw_row in mds[data_start_idx:]:
        row = raw_row.rstrip()
        if not row.startswith("|"):
            # Skip lines that aren't table rows
            continue
        row_with_nl = row + "\n"
        row_tokens = get_token_count(row_with_nl)

        # If adding this row would exceed n_tokens, finalize the current chunk
        if current_token_count + row_tokens > n_tokens:
            # Prepend the header to the collected rows and append to chunks
            chunk_text = header + "".join(current_chunk_data)
            chunks.append(chunk_text)

            # Build overlap buffer by walking backwards over the data rows
            overlap_buffer = []
            overlap_tokens = 0
            for prev_row_with_nl in reversed(current_chunk_data):
                prev_tokens = get_token_count(prev_row_with_nl)
                if overlap_tokens + prev_tokens > overlap:
                    break
                overlap_buffer.insert(0, prev_row_with_nl)
                overlap_tokens += prev_tokens

            # Start a new chunk with just the overlap rows
            current_chunk_data = list(overlap_buffer)
            current_token_count = header_token_count + overlap_tokens

        # Add this row to the (new or continuing) chunk
        current_chunk_data.append(row_with_nl)
        current_token_count += row_tokens

    # If any rows remain, append them as the final chunk
    if current_chunk_data:
        chunk_text = header + "".join(current_chunk_data)
        chunks.append(chunk_text)

    return chunks, header

def chunk_markdown_table(
        prompt: str,
        md_table: str,
        cols: List[str],
        n_tokens: int,
        overlap: int,
        ollama_client: ollama.Client,
        model: str,
        model_options: Dict
):
    """
    Splits a markdown table into chunks with overlapping tokens and generates a summary of the table.

    Args:
        - prompt (str): The prompt to use for the model.
        - md_table (str): The markdown table as a string.
        - cols (List[str]): List of column headers. If provided, overrides the header in md_table.
        - n_tokens (int): Maximum number of tokens per chunk (including header tokens).
        - overlap (int): Number of tokens to overlap between adjacent chunks (data‐rows only).
        - ollama_client (ollama_client): The Ollama client to use for querying the model.
        - model (str): The name of the model to use for chunking.
        - model_options (Dict): A dictionary containing options for the model, such as temperature.

    Returns:
        - chunks (List[str]): Each element is a string consisting of (header + data‐rows with overlaps).
        - header (str): The header block (header row + separator row).
        - summary (str): A summary of the table generated by the model.
    """
    prompt = prompt.format(
        table=md_table.split("\n")
    )
    output = ask_LLM_with_JSON(prompt, ollama_client, model, model_options)
    
    try:
        outd = recover_json(output)
        cols = outd["columns"].split(",")
        summary = outd["summary_of_the_table"]
    except:
        print(f"Could not recover with malformed JSON {output}")
        # logc(f"Could not recover with malformed JSON {output}")
        return [], "", ""

    chunks, header = chunk_markdown_table_with_overlap(
        md_table, cols, n_tokens=n_tokens, overlap=overlap
    )
    print("Chunks:", len(chunks))

    return chunks, header, summary

def chunk_table_data(
    prompt: str,
    tables_dir: str,
    cols: List[List[str]],
    n_tokens: int,
    overlap: int,
    ollama_client: ollama.Client,
    model: str,
    model_options: Dict,
    tables_chunks_dir: str,
) -> None:
    """
    Opens markdown tables and split them into chunnks, then, stored the resulting tables chunks as well as summaries in dedicated folders.

    Args:
        - prompt (str): The prompt to use for the model.
        - tables_dir (str): The path to the directory where to find the tables.
        - cols (List[List[str]]): A list of lists of columns corresponding to the tables respectively.
        - n_token (int): The chunk size.
        - overlap (int): Number of tokens to overlap between adjacent chunks (data‐rows only).
        - model (str): The name of the model to use for chunking.
        - ollama_client (Ollama Client): The Ollama client to use for querying the model.
        - model_options (Dict): A dictionary containing the options (like the temperature) to run the model.
        - tables_chunks_dir (str): The path to the directory where to stored the generated tables chunks.

    Returns:
        None: The function saves the chunks and summaries to the specified directories.
    """
    tables_path = Path(tables_dir)
    tables_chunks_path = Path(tables_chunks_dir)

    tables_chunks_path.mkdir(parents=True, exist_ok=True)

    if not tables_path.is_dir():
        raise ValueError(f"Folder '{tables_path}' does not exist.")

    print(f"\nProcessing the tables in {tables_path}...\n")

    if not cols:
        for table_path in tables_path.rglob("*_Table_*.md"):
            if table_path.is_file():
                print(f"\nReading file: {table_path}")
                with open(table_path, "r", encoding="utf-8") as f:
                    md_table = f.read()
                    chunks, _, summary = chunk_markdown_table(
                        prompt, md_table, None, n_tokens, overlap, ollama_client, model, model_options
                    )

                    for idx, chunk in enumerate(chunks, 1):
                        chunk_with_summary = f"{summary}\n\n{chunk}"
                        table_name = (
                            tables_chunks_path / f"{table_path.stem}_chunk_{idx}.md"
                        )
                        with open(table_name, "w", encoding="utf-8") as file:
                            file.write(chunk_with_summary)
                        print(f"Saved table chunk to: {table_name}")

    elif len(cols) == len(list(tables_path.rglob("*_Table_*.md"))):
        for idx, table_path in enumerate(tables_path.rglob("*.md")):
            if table_path.is_file():
                print(f"\nReading file: {table_path}")
                with open(table_path, "r", encoding="utf-8") as f:
                    md_table = f.read()
                    chunks, _, summary = chunk_markdown_table(
                        prompt, md_table, cols[idx], n_tokens, overlap, ollama_client, model, model_options
                    )

                    for idxx, chunk in enumerate(chunks, 1):
                        chunk_with_summary = f"{summary}\n\n{chunk}"
                        table_name = (
                            tables_chunks_path / f"{table_path.stem}_chunk_{idxx}.md"
                        )
                        with open(table_name, "w", encoding="utf-8") as file:
                            file.write(chunk_with_summary)
                        print(f"Saved table to: {table_name}")

    else:
        print("Please provide a list of columns for each table.")

    print(
        f"\nAll tables have been processed.\nTables chunks were saved in Markdown to: {tables_dir}."
    )


