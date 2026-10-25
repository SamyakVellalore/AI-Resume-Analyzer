# Resume Intelligence System

An educational resume-analysis prototype that extracts text from PDF/DOCX resumes, identifies skills from a curated dictionary, ranks six example job roles using TF-IDF and cosine similarity, and provides skill-gap feedback and a basic learning roadmap.

## Features
- PDF and DOCX upload
- Resume text extraction and preview
- Categorized keyword-based skill detection
- TF-IDF + cosine-similarity role ranking
- Top-three role recommendations
- Selected-role skill-gap analysis
- Rule-based learning roadmap
- Downloadable CSV and TXT reports

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   streamlit run app.py
   ```

## Project method
The application uses keyword matching for skill extraction, TF-IDF vectors and cosine similarity for role ranking, set comparisons for skill gaps, and a rule-based roadmap. Similarity scores represent textual overlap, not the probability of getting hired.

## Limitations and responsible use
- Six curated role profiles are examples, not a live job-market dataset.
- Keyword matching may miss synonyms or skills described indirectly.
- Scanned/image-only PDFs may require OCR.
- Do not use this prototype as an automated hiring decision system.
- Resume data may contain personal information. Use test/sample resumes where possible and do not retain or share files without permission.

## Testing checklist
- [ ] Upload a text-based PDF resume.
- [ ] Upload a DOCX resume.
- [ ] Check behavior for an unsupported file type.
- [ ] Check behavior for a scanned PDF or a file with no extractable text.
- [ ] Verify that detected skills are present in the resume.
- [ ] Check whether adding/removing a relevant skill changes role ranking sensibly.
- [ ] Confirm skill-gap and report downloads work.
