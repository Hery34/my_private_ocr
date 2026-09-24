from fastapi import FastAPI, UploadFile
from paddleocr import PaddleOCR
import tempfile, os
import fitz  # PyMuPDF

app = FastAPI(title="PaddleOCR API")

# Langue OCR : "fr", "en", "ch"...
ocr = PaddleOCR(use_angle_cls=True, lang="fr", show_log=False)


def ocr_image_path(path):
    result = ocr.ocr(path, cls=True)
    return [
        {"texte": mot[1][0], "score": round(float(mot[1][1]), 4)}
        for page in result if page for mot in page
    ]


def ocr_pdf_bytes(data, dpi=200):
    pages_out = []
    doc = fitz.open(stream=data, filetype="pdf")
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            tmp_path = tmp.name
        try:
            lignes = ocr_image_path(tmp_path)
        finally:
            os.remove(tmp_path)
        pages_out.append({"page": i + 1, "lignes": lignes})
    doc.close()
    return pages_out


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(file: UploadFile):
    data = await file.read()
    filename = (file.filename or "").lower()
    is_pdf = filename.endswith(".pdf") or file.content_type == "application/pdf"

    if is_pdf:
        pages = ocr_pdf_bytes(data)
        total = sum(len(p["lignes"]) for p in pages)
        return {"type": "pdf", "nb_pages": len(pages),
                "nb_lignes": total, "pages": pages}

    # sinon : image
    suffix = os.path.splitext(filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        lignes = ocr_image_path(path)
        return {"type": "image", "nb_lignes": len(lignes), "resultat": lignes}
    finally:
        os.remove(path)
