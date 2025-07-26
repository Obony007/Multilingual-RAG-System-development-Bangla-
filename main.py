from src.rag import RAGSystem

def main():
    rag = RAGSystem(pdf_path='data/HSC26-Bangla1st-Paper.pdf')
    
    test_cases = [
        ("অনুপমের ভাষায় সুপুরুষ কাকে কে বলা হয়েছে?", "শুম্ভুনাথ"),
        ("কাকে কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর")
    ]
    
    results = rag.evaluate_retrieval(test_cases)
    
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Expected: {result['expected']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Grounded: {result['grounded']}")
        print("-" * 50)

if __name__ == "__main__":
    main()