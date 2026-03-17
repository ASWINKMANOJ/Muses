from app.ingestion.loaders.pdf_parser import extract_sections_from_pdf

sections = extract_sections_from_pdf("data/sample.pdf")

print(f"Total sections: {len(sections)}\n")

for i, sec in enumerate(sections[:3]):
    print(f"--- Section {i} ---")
    print("Heading:", sec["heading"])
    print("Text preview:", sec["text"][:200])
    print()