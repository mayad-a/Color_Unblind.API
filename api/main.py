# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
from daltonlens import simulate

app = FastAPI()
@app.get("/")
def root():
    return {"message": "FastAPI is running on Vercel!"}

# Allow CORS (تعدلي القائمة لو عايزة دومين محدد بدل "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ربط الأرقام بأنواع عمى الألوان
cvd_types = {
    1: simulate.Deficiency.PROTAN,  # protanopia
    2: simulate.Deficiency.DEUTAN,  # deuteranopia
    3: simulate.Deficiency.TRITAN,  # tritanopia
}

simulator = simulate.Simulator_Machado2009()


def daltonize_simple(original, simulated):
    error = original.astype(int) - simulated.astype(int)
    corrected = original.astype(int) + error
    corrected = np.clip(corrected, 0, 255).astype("uint8")
    return corrected


@app.post("/correct")
async def correct_image(
        file: UploadFile = File(...),
        cvd_type: int = Form(...)
):
    if cvd_type not in cvd_types:
        return {"error": "Invalid cvd_type. Use 1=protanopia, 2=deuteranopia, 3=tritanopia"}

    img = Image.open(file.file).convert("RGB")
    img_np = np.asarray(img)

    simulated = simulator.simulate_cvd(img_np, cvd_types[cvd_type], severity=1.0)
    corrected = daltonize_simple(img_np, simulated)
    corrected_img = Image.fromarray(corrected.astype("uint8"))

    buf = io.BytesIO()
    corrected_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
