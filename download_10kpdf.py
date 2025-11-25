from dotenv import load_dotenv
import os
from sec_api import PdfGeneratorApi
load_dotenv()
#API to download PDFs
api_key = os.getenv("PDF_API_KEY")

pdfGeneratorApi = PdfGeneratorApi(api_key)

# example URL of Tesla's 2024 10-K filing
filing_10K_url = "https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm"

# download 10-K filing as PDF
pdf_10K_filing = pdfGeneratorApi.get_pdf(filing_10K_url)

# save PDF of 10-K filing to disk
with open("dataset/tesla_10K.pdf", "wb") as file:
    file.write(pdf_10K_filing)