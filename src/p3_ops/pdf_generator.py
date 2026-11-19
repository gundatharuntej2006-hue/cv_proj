from pathlib import Path
import hashlib
import os
from p3_ops._tbcore_adapter import legacy_schema

PipelineResult = legacy_schema("PipelineResult")


class TriagePDFReportGenerator:
    def generate_report(self, result: PipelineResult, output_path: str = "report/triage_report.pdf") -> str:
        if os.environ.get("TB_MOCK") != "1":
            raise RuntimeError("Legacy PipelineResult lacks C14 provenance; use p3_ops.api.make_pdf with a real CaseResult")
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from ._tbcore_adapter import disclaimer_texts
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cid = hashlib.sha256(str(result.image_id).encode()).hexdigest()[:12]
        disclaimer, _ = disclaimer_texts()
        c = canvas.Canvas(str(out), pagesize=A4)
        c.setTitle("Synthetic legacy demonstration")
        c.drawString(40, 790, "SYNTHETIC demonstration - no measured model result")
        c.drawString(40, 765, f"Case {cid}")
        text = c.beginText(40, 720)
        import textwrap
        for line in textwrap.wrap(disclaimer, 90):
            text.textLine(line)
        c.drawText(text)
        c.showPage()
        c.save()
        return str(out)
