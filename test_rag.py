# test_rag_stream.py

from app.pipeline.query_pipeline import query_pipeline_stream

while True:
    query = input("\nAsk: ")

    print("\nAnswer:\n")

    for token in query_pipeline_stream(query):
        print(token, end="", flush=True)

    print("\n")