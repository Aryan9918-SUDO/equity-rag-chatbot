import requests
import fitz
import time

BASE_URL = "http://127.0.0.1:5000"

def create_sample_esop_pdf(filename="Sample_ESOP_Plan.pdf"):
    """
    Generates a realistic dummy ESOP PDF programmatically.
    This ensures you have a pristine, valid PDF with known answers for the test script.
    """
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "ACME CORP EMPLOYEE STOCK OWNERSHIP PLAN (ESOP)\n\n"
        "1. Eligibility (Who is eligible for ESOPs?)\n"
        "All full-time employees who have completed at least 6 months of continuous service are dynamically eligible to participate in the ESOP program.\n\n"
        "2. Vesting (What is the vesting schedule?)\n"
        "The standard vesting schedule is a 4-year period. Shares vest at a rate of 25% after the first year, and then equally every month thereafter until fully vested.\n\n"
        "3. Cliff Period (What is the cliff period?)\n"
        "The cliff period is strictly set to 12 months from the vesting commencement date. No shares will vest before this cliff is officially reached.\n\n"
        "4. Termination (What happens to shares if an employee leaves?)\n"
        "If an employee leaves voluntarily or is terminated without cause, they retain all vested shares. Any unvested shares are immediately forfeited and returned to the option pool.\n\n"
        "5. Pricing (What is the exercise price?)\n"
        "The exercise price is pegged to the Fair Market Value (FMV) of the company's common stock on the exact date of the grant, as determined by the Board of Directors via independent 409A valuation.\n"
    )
    # Insert text into the designated box shape
    page.insert_textbox(fitz.Rect(50, 50, 500, 750), text, fontsize=12)
    doc.save(filename)
    doc.close()
    return filename

def test_pipeline():
    print("--- 1. Testing /health endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection failed. Please ensure the Flask server is running in another terminal!")
        return

    print("\n--- 2. Generating & Uploading Test PDF ---")
    pdf_filename = create_sample_esop_pdf()
    print(f"Local test file created: {pdf_filename}")
    
    with open(pdf_filename, 'rb') as f:
        files = {'file': (pdf_filename, f, 'application/pdf')}
        print("Uploading & building vector store (this might take a few seconds)...")
        start = time.time()
        
        up_response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"HTTP {up_response.status_code} - Body: {up_response.json()}")
        print(f"Ingestion took {round(time.time() - start, 2)} seconds")

    if up_response.status_code != 200:
        print("Upload failed. Halting tests.")
        return

    print("\n--- 3. Asking Trial Questions ---")
    questions = [
        "What is the vesting schedule?",
        "What happens to shares if an employee leaves?",
        "What is the cliff period?",
        "Who is eligible for ESOPs?",
        "What is the exercise price?"
    ]

    for idx, q in enumerate(questions, 1):
        print(f"\nQ{idx}: {q}")
        req = requests.post(f"{BASE_URL}/ask", json={"question": q})
        
        if req.status_code == 200:
            data = req.json()
            print(f"A: {data.get('answer')}")
            print(f"Source Pages: {data.get('sources')}")
            metadata = data.get('source_metadata', [])
            if metadata:
                print(f"Source Metadata:")
                for m in metadata:
                    print(f"  - Page {m['page']} ({m['source']}): \"{m['chunk_preview']}\"")
            else:
                print(f"Source Metadata: (none)")
        else:
            print(f"Query Failed (HTTP {req.status_code}): {req.text}")
            
    print("\n--- Testing Complete! ---")

if __name__ == "__main__":
    test_pipeline()
