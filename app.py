"""
NicheReport.ai — Unified Automation Service
============================================
Handles the full pipeline autonomously:
  1. Receives Gumroad sale webhook (POST /webhook/gumroad)
  2. Extracts buyer info and niche topic from the sale
  3. Generates a full market research report via GPT-4o
  4. Renders it as a professional PDF
  5. Uploads the PDF to a temporary URL (or attaches inline)
  6. Emails the PDF download link to the buyer via Resend
  7. Returns 200 OK to Gumroad

Environment variables required:
  OPENAI_API_KEY   — OpenAI API key (sk-proj-...)
  RESEND_API_KEY   — Resend API key (re_...)
  FROM_EMAIL       — Sender email (reports@nichereport.ai)
"""

import os
import io
import json
import logging
import hashlib
import time
from datetime import datetime
from flask import Flask, request, jsonify
import openai
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "reports@nichereport.ai")

# ─── Colors ───────────────────────────────────────────────────────────────────
DARK_BG     = HexColor("#1a1a2e")
GOLD        = HexColor("#c9a84c")
CREAM       = HexColor("#f5f0e8")
DARK_GRAY   = HexColor("#2d2d44")
LIGHT_GRAY  = HexColor("#f0ede6")
MID_GRAY    = HexColor("#888888")
TEXT_DARK   = HexColor("#1a1a2e")

# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "NicheReport.ai service running", "version": "2.0"})

@app.route("/health", methods=["GET"])
def health2():
    return jsonify({"status": "ok"})

# ─── Gumroad Webhook ──────────────────────────────────────────────────────────
@app.route("/webhook/gumroad", methods=["POST"])
def gumroad_webhook():
    """
    Gumroad sends a POST with form-encoded data on every sale.
    Fields include: email, full_name, product_name, product_permalink,
                    price, sale_id, variants (JSON string with custom fields)
    """
    try:
        data = request.form.to_dict()
        logger.info(f"Gumroad webhook received: {json.dumps(data, indent=2)}")

        buyer_email = data.get("email", "")
        buyer_name  = data.get("full_name", "Valued Customer")
        product_name = data.get("product_name", "Market Research Report")
        sale_id     = data.get("sale_id", "unknown")

        # Extract niche topic from custom fields or product name
        # Gumroad sends custom fields as JSON in 'variants' or individual fields
        niche_topic = extract_niche_topic(data, product_name)

        if not buyer_email:
            logger.error("No buyer email in webhook data")
            return jsonify({"error": "No buyer email"}), 400

        logger.info(f"Processing order for {buyer_email} — niche: {niche_topic}")

        # Generate report content via GPT-4o
        report_content = generate_report_content(niche_topic)

        # Render PDF
        pdf_bytes = render_pdf(niche_topic, report_content, buyer_name)

        # Upload PDF to file.io (free, no auth needed, 14-day link)
        pdf_url = upload_pdf(pdf_bytes, niche_topic, sale_id)

        # Send delivery email via Resend
        send_delivery_email(buyer_email, buyer_name, niche_topic, pdf_url)

        logger.info(f"✅ Report delivered to {buyer_email} for niche: {niche_topic}")
        return jsonify({"status": "success", "message": "Report generated and delivered"}), 200

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def extract_niche_topic(data: dict, product_name: str) -> str:
    """Extract the niche topic from Gumroad webhook data."""
    # Try custom fields first (Gumroad sends them as 'custom_fields' JSON)
    custom_fields_raw = data.get("custom_fields", "")
    if custom_fields_raw:
        try:
            fields = json.loads(custom_fields_raw)
            for field in fields:
                if "niche" in field.get("name", "").lower() or "topic" in field.get("name", "").lower():
                    return field.get("value", "")
        except Exception:
            pass

    # Try individual field keys
    for key in ["niche", "topic", "niche_topic", "market", "industry"]:
        if data.get(key):
            return data[key]

    # Fall back to product name (e.g. "Deep Dive Report: Pet Supplements")
    if ":" in product_name:
        return product_name.split(":", 1)[1].strip()

    return product_name


def generate_report_content(niche_topic: str) -> dict:
    """Call GPT-4o to generate a full market research report."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are a senior market research analyst. Generate a comprehensive, data-rich market research report for the following niche:

NICHE: {niche_topic}

Your report must include ALL of the following sections with substantial, specific content:

1. EXECUTIVE_SUMMARY (3-4 paragraphs with key findings and market opportunity)
2. MARKET_OVERVIEW (market size, growth rate, key drivers, 2024-2028 outlook)
3. TARGET_AUDIENCE (3 detailed buyer personas with demographics, psychographics, pain points, spending habits)
4. COMPETITOR_ANALYSIS (5 key competitors with their strengths, weaknesses, pricing, market share estimate)
5. MARKET_GAPS (3-5 specific underserved opportunities with evidence)
6. MONETIZATION_STRATEGIES (5 specific revenue models with realistic revenue projections)
7. CONTENT_MARKETING (10 high-value content topics with estimated search volume and intent)
8. TRAFFIC_CHANNELS (top 5 acquisition channels with specific tactics and expected ROI)
9. TOOLS_AND_RESOURCES (10 essential tools with pricing and use case)
10. ACTION_PLAN (30/60/90 day roadmap with specific milestones and KPIs)

Return ONLY a valid JSON object with these exact keys (no markdown, no code blocks):
{{
  "executive_summary": "...",
  "market_overview": "...",
  "target_audience": "...",
  "competitor_analysis": "...",
  "market_gaps": "...",
  "monetization_strategies": "...",
  "content_marketing": "...",
  "traffic_channels": "...",
  "tools_and_resources": "...",
  "action_plan": "..."
}}

Each value should be 200-400 words of specific, actionable, data-backed content. Include specific numbers, percentages, dollar amounts, and named companies/tools wherever possible."""

    logger.info(f"Calling GPT-4o for niche: {niche_topic}")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=6000
    )

    content_str = response.choices[0].message.content.strip()
    # Strip markdown code blocks if present
    if content_str.startswith("```"):
        content_str = content_str.split("```")[1]
        if content_str.startswith("json"):
            content_str = content_str[4:]
    content_str = content_str.strip()

    return json.loads(content_str)


def render_pdf(niche_topic: str, content: dict, buyer_name: str) -> bytes:
    """Render a professional PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title=f"Market Research Report: {niche_topic}",
        author="NicheReport.ai"
    )

    styles = getSampleStyleSheet()

    # Custom styles
    cover_title = ParagraphStyle(
        'CoverTitle', fontSize=28, fontName='Helvetica-Bold',
        textColor=white, alignment=TA_CENTER, spaceAfter=12, leading=34
    )
    cover_sub = ParagraphStyle(
        'CoverSub', fontSize=14, fontName='Helvetica',
        textColor=GOLD, alignment=TA_CENTER, spaceAfter=8
    )
    cover_meta = ParagraphStyle(
        'CoverMeta', fontSize=11, fontName='Helvetica',
        textColor=CREAM, alignment=TA_CENTER, spaceAfter=6
    )
    section_heading = ParagraphStyle(
        'SectionHeading', fontSize=16, fontName='Helvetica-Bold',
        textColor=DARK_BG, spaceBefore=20, spaceAfter=10,
        borderPad=4
    )
    body_text = ParagraphStyle(
        'BodyText2', fontSize=10.5, fontName='Helvetica',
        textColor=TEXT_DARK, alignment=TA_JUSTIFY,
        spaceAfter=8, leading=16
    )
    toc_style = ParagraphStyle(
        'TOC', fontSize=11, fontName='Helvetica',
        textColor=TEXT_DARK, spaceAfter=6, leftIndent=20
    )

    story = []

    # ── Cover Page ──────────────────────────────────────────────────────────
    # Dark background cover using a table
    cover_data = [[
        Paragraph(f"MARKET RESEARCH REPORT", cover_title),
    ]]
    cover_table = Table(cover_data, colWidths=[7*inch])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 60),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 30),
        ('RIGHTPADDING', (0,0), (-1,-1), 30),
    ]))
    story.append(cover_table)

    niche_data = [[Paragraph(niche_topic.upper(), cover_sub)]]
    niche_table = Table(niche_data, colWidths=[7*inch])
    niche_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(niche_table)

    meta_info = [
        [Paragraph(f"Prepared for: {buyer_name}", cover_meta)],
        [Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", cover_meta)],
        [Paragraph("Powered by NicheReport.ai", cover_meta)],
    ]
    meta_table = Table(meta_info, colWidths=[7*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)

    # Gold divider bar
    divider_data = [[""]]
    divider_table = Table(divider_data, colWidths=[7*inch], rowHeights=[8])
    divider_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), GOLD),
    ]))
    story.append(divider_table)
    story.append(PageBreak())

    # ── Table of Contents ───────────────────────────────────────────────────
    story.append(Paragraph("TABLE OF CONTENTS", section_heading))
    story.append(HRFlowable(width="100%", thickness=2, color=GOLD))
    story.append(Spacer(1, 12))

    sections = [
        ("1", "Executive Summary"),
        ("2", "Market Overview"),
        ("3", "Target Audience & Buyer Personas"),
        ("4", "Competitor Analysis"),
        ("5", "Market Gaps & Opportunities"),
        ("6", "Monetization Strategies"),
        ("7", "Content Marketing Blueprint"),
        ("8", "Traffic & Acquisition Channels"),
        ("9", "Tools & Resources"),
        ("10", "90-Day Action Plan"),
    ]
    for num, title in sections:
        story.append(Paragraph(f"{num}.  {title}", toc_style))
    story.append(PageBreak())

    # ── Content Sections ────────────────────────────────────────────────────
    section_map = [
        ("1. Executive Summary", "executive_summary"),
        ("2. Market Overview", "market_overview"),
        ("3. Target Audience & Buyer Personas", "target_audience"),
        ("4. Competitor Analysis", "competitor_analysis"),
        ("5. Market Gaps & Opportunities", "market_gaps"),
        ("6. Monetization Strategies", "monetization_strategies"),
        ("7. Content Marketing Blueprint", "content_marketing"),
        ("8. Traffic & Acquisition Channels", "traffic_channels"),
        ("9. Tools & Resources", "tools_and_resources"),
        ("10. 90-Day Action Plan", "action_plan"),
    ]

    for i, (title, key) in enumerate(section_map):
        story.append(Paragraph(title, section_heading))
        story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
        story.append(Spacer(1, 8))

        section_text = content.get(key, "Content not available.")
        # Split into paragraphs on double newlines
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [section_text]

        for para in paragraphs:
            # Handle bullet-style lines
            lines = para.split("\n")
            if len(lines) > 1:
                for line in lines:
                    line = line.strip()
                    if line.startswith(("•", "-", "*", "–")):
                        line = "• " + line.lstrip("•-*– ")
                    if line:
                        story.append(Paragraph(line, body_text))
            else:
                story.append(Paragraph(para, body_text))

        story.append(Spacer(1, 12))

        # Page break after every 2 sections except the last
        if i < len(section_map) - 1 and (i + 1) % 2 == 0:
            story.append(PageBreak())

    # ── Footer Page ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    footer_data = [[
        Paragraph("Thank you for your purchase.", cover_sub),
    ], [
        Paragraph("NicheReport.ai — AI-Powered Market Intelligence", cover_meta),
    ], [
        Paragraph("For questions, reply to your delivery email.", cover_meta),
    ]]
    footer_table = Table(footer_data, colWidths=[7*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 40),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(footer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def upload_pdf(pdf_bytes: bytes, niche_topic: str, sale_id: str) -> str:
    """Upload PDF to file.io and return a download URL."""
    safe_name = niche_topic.lower().replace(" ", "_")[:40]
    filename = f"nichereport_{safe_name}_{sale_id[:8]}.pdf"

    try:
        resp = requests.post(
            "https://file.io/?expires=14d",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=30
        )
        data = resp.json()
        if data.get("success"):
            url = data.get("link", "")
            logger.info(f"PDF uploaded to file.io: {url}")
            return url
    except Exception as e:
        logger.warning(f"file.io upload failed: {e}, trying 0x0.st...")

    # Fallback: 0x0.st (no expiry, permanent)
    try:
        resp = requests.post(
            "https://0x0.st",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=30
        )
        url = resp.text.strip()
        logger.info(f"PDF uploaded to 0x0.st: {url}")
        return url
    except Exception as e:
        logger.error(f"All upload methods failed: {e}")
        return ""


def send_delivery_email(buyer_email: str, buyer_name: str, niche_topic: str, pdf_url: str):
    """Send the report delivery email via Resend."""
    first_name = buyer_name.split()[0] if buyer_name else "there"

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: Georgia, serif; background: #f5f0e8; margin: 0; padding: 0;">
      <div style="max-width: 600px; margin: 0 auto; background: white;">

        <!-- Header -->
        <div style="background: #1a1a2e; padding: 40px 40px 30px; text-align: center;">
          <h1 style="color: #c9a84c; font-size: 24px; margin: 0 0 8px; letter-spacing: 2px;">NICHEREPORT.AI</h1>
          <p style="color: #f5f0e8; margin: 0; font-size: 14px;">AI-Powered Market Intelligence</p>
        </div>

        <!-- Body -->
        <div style="padding: 40px;">
          <h2 style="color: #1a1a2e; font-size: 22px; margin-bottom: 16px;">
            Your Report is Ready, {first_name}!
          </h2>
          <p style="color: #333; font-size: 15px; line-height: 1.7;">
            Your custom market research report on <strong>{niche_topic}</strong> has been generated
            and is ready for download. This report includes:
          </p>
          <ul style="color: #333; font-size: 14px; line-height: 2;">
            <li>Market size, growth trends &amp; 5-year outlook</li>
            <li>3 detailed buyer personas</li>
            <li>Competitor analysis (5 key players)</li>
            <li>Market gaps &amp; monetization strategies</li>
            <li>Content marketing blueprint &amp; traffic channels</li>
            <li>90-day action plan with KPIs</li>
          </ul>

          <div style="text-align: center; margin: 32px 0;">
            <a href="{pdf_url}"
               style="background: #1a1a2e; color: #c9a84c; padding: 16px 40px;
                      text-decoration: none; border-radius: 4px; font-size: 16px;
                      font-weight: bold; display: inline-block; letter-spacing: 1px;">
              DOWNLOAD YOUR REPORT →
            </a>
          </div>

          <p style="color: #666; font-size: 13px; line-height: 1.6;">
            <strong>Note:</strong> Your download link is valid for 14 days. We recommend saving
            the PDF to your device immediately.
          </p>
        </div>

        <!-- Footer -->
        <div style="background: #1a1a2e; padding: 24px 40px; text-align: center;">
          <p style="color: #888; font-size: 12px; margin: 0;">
            © {datetime.now().year} NicheReport.ai · Reply to this email for support
          </p>
        </div>
      </div>
    </body>
    </html>
    """

    payload = {
        "from": f"NicheReport.ai <{FROM_EMAIL}>",
        "to": [buyer_email],
        "subject": f"Your Market Research Report is Ready: {niche_topic}",
        "html": html_body
    }

    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=15
    )

    if resp.status_code in (200, 201):
        logger.info(f"✅ Delivery email sent to {buyer_email}")
    else:
        logger.error(f"❌ Resend failed: {resp.status_code} — {resp.text}")
        raise Exception(f"Email delivery failed: {resp.text}")


# ─── Test endpoint ────────────────────────────────────────────────────────────
@app.route("/test", methods=["POST"])
def test_pipeline():
    """Test the full pipeline with a fake sale."""
    data = request.json or {}
    buyer_email = data.get("email", "test@example.com")
    buyer_name  = data.get("name", "Test User")
    niche_topic = data.get("niche", "AI Productivity Tools for Freelancers")

    try:
        report_content = generate_report_content(niche_topic)
        pdf_bytes = render_pdf(niche_topic, report_content, buyer_name)
        pdf_url = upload_pdf(pdf_bytes, niche_topic, "test123")
        send_delivery_email(buyer_email, buyer_name, niche_topic, pdf_url)
        return jsonify({"status": "success", "pdf_url": pdf_url, "email_sent_to": buyer_email}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
