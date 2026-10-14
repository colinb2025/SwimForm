"""
SwimForm web app.

  uvicorn app.main:app --reload

GET  /                   upload page: two clips, where to put the camera for each
POST /analyses           run the analysis; on a clip problem, back to the form with the fix
GET  /analyses/{id}      the report
"""
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from swimform import rules as R
from swimform.analyze import DEFAULT_RULES

from . import pipeline, presenter

HERE = Path(__file__).resolve().parent
app = FastAPI(title="SwimForm")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

_rules_cache = {}


def rules():
    if "r" not in _rules_cache:
        r, errors = R.load(DEFAULT_RULES)
        if errors:
            raise RuntimeError("rules workbook invalid: " + "; ".join(errors))
        _rules_cache["r"] = r
    return _rules_cache["r"]


@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {"problems": {}})


@app.post("/analyses")
async def create_analysis(request: Request, side: UploadFile | None = None, head_on: UploadFile | None = None):
    files = {}
    for slot, up in (("side", side), ("head_on", head_on)):
        if up is not None and up.filename:
            files[slot] = (up.filename, up.file)
    try:
        aid = pipeline.run(files)
    except pipeline.UploadProblem as e:
        return templates.TemplateResponse(request, "upload.html", {"problems": e.problems}, status_code=422)
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


@app.get("/analyses/{aid}", response_class=HTMLResponse)
def report_page(request: Request, aid: str):
    record = pipeline.load(aid)
    if record is None:
        return templates.TemplateResponse(request, "missing.html", {}, status_code=404)
    return templates.TemplateResponse(request, "report.html", {"r": presenter.build(record, rules())})


@app.get("/healthz")
def healthz():
    return {"ok": True}
