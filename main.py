from fastapi import FastAPI, UploadFile
from paddleocr import PaddleOCR
import tempfile, os

app = FastAPI(title="PaddleOCR API")

# Changez lang ici : "fr", "en", "ch", "japan", "korean"...
ocr = PaddleOCR(use_angle_cls=True, lang="fr", show_log=False)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ocr")
async def run_ocr(file: UploadFile):
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        result = ocr.ocr(path, cls=True)
        lignes = [
            {"texte": mot[1][0], "score": round(float(mot[1][1]), 4)}
            for page in result if page for mot in page
        ]
        return {"nb_lignes": len(lignes), "resultat": lignes}
    finally:
        os.remove(path)