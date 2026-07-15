"""Generates corpus/sla_support_policy.pdf. Run once; the output is what's
actually ingested. Kept here so the PDF's content is auditable/regenerable
rather than an opaque binary with no source."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceAfter=8, spaceBefore=12)
body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=8, leading=14)

doc = SimpleDocTemplate("corpus/sla_support_policy.pdf", pagesize=LETTER,
                         topMargin=0.8 * inch, bottomMargin=0.8 * inch)

story = [
    Paragraph("SLA and support policy", h1),
    Paragraph(
        "This document describes NimbusStore's uptime commitment and support "
        "response times by account tier.",
        body,
    ),
    Paragraph("Uptime SLA", h2),
    Paragraph(
        "Pro and Enterprise tier accounts have a formal uptime commitment of "
        "99.9% monthly uptime. The Starter tier has no formal uptime SLA and "
        "is provided on a best-effort basis. The Free tier has no uptime "
        "commitment of any kind.",
        body,
    ),
    Paragraph("Service credits", h2),
    Paragraph(
        "For Pro and Enterprise accounts, if monthly uptime falls between "
        "99.0% and 99.9%, customers receive a service credit equal to 10% of "
        "that month's bill. If monthly uptime falls below 99.0%, the credit "
        "increases to 25% of that month's bill. Credits are capped at 100% "
        "of the affected month's bill and must be requested within 30 days.",
        body,
    ),
    Paragraph("Support response times", h2),
    Paragraph(
        "For Severity 1 issues (complete service outage), Enterprise "
        "customers receive a 1 hour response time commitment, and Pro "
        "customers receive a 4 hour response time commitment. Starter tier "
        "customers are supported on a next-business-day basis through the "
        "community forum only, with no guaranteed response time.",
        body,
    ),
    Paragraph("Support channels", h2),
    Paragraph(
        "Email support is available to all paid tiers. In-app chat support "
        "is available to Pro and Enterprise. A dedicated Slack channel with "
        "the NimbusStore support team is available to Enterprise customers "
        "only.",
        body,
    ),
]

doc.build(story)
print("wrote corpus/sla_support_policy.pdf")
