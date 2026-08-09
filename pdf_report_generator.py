from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import os


def generate_pdf_report(inspection, output_path):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(output_path)

    story = []

    # ---------------------------------------------------
    # Title
    # ---------------------------------------------------

    story.append(
        Paragraph(
            "<b><font size=20>SAFESIGHT AI Inspection Report</font></b>",
            styles["Title"],
        )
    )

    story.append(Spacer(1, 20))

    # ---------------------------------------------------
    # Inspection Details
    # ---------------------------------------------------

    story.append(
        Paragraph(
            f"<b>Inspection ID:</b> {inspection['_id']}",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Date:</b> {inspection['timestamp']}",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 15))

    # ---------------------------------------------------
    # Overall Summary
    # ---------------------------------------------------

    summary = inspection["summary"]

    table_data = [
        ["Metric", "Value"],

        ["Total Images", summary["total_images"]],

        ["Total Workers", summary["total_workers"]],

        ["Safe Workers", summary["safe_workers"]],

        ["Unsafe Workers", summary["unsafe_workers"]],

        ["Overall Compliance",
         f"{summary['compliance']}%"],

        ["Missing Helmets", summary["helmet"]],

        ["Missing Vests", summary["vest"]],

        ["Missing Gloves", summary["gloves"]],

        ["Missing Boots", summary["boots"]],

        ["Missing Goggles", summary["goggles"]],
    ]

    table = Table(table_data)

    table.setStyle(
        TableStyle(
            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 1, colors.black),

                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),

                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    story.append(table)

    story.append(Spacer(1, 25))

    # ---------------------------------------------------
    # Individual Images
    # ---------------------------------------------------

    for image in inspection["images"]:

        story.append(
            Paragraph(
                f"<b><font size=16>{image['name']}</font></b>",
                styles["Heading1"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Site:</b> {image['site_name']}",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 10))

        # ---------------------------------------------------
        # Annotated Image
        # ---------------------------------------------------

        image_path = image["annotated"].replace(
            "http://127.0.0.1:8000/",
            ""
        )

        if os.path.exists(image_path):

            story.append(
                Image(
                    image_path,
                    width=5.5 * inch,
                    height=4 * inch,
                )
            )

            story.append(Spacer(1, 15))

        # ---------------------------------------------------
        # Worker Statistics
        # ---------------------------------------------------

        stats = [
            ["Workers", image["total_workers"]],

            ["Safe", image["safe_workers"]],

            ["Unsafe", image["unsafe_workers"]],

            ["Compliance",
             f"{image['compliance_rate']}%"],

            ["Status", image["status"]],
        ]

        worker_table = Table(stats)

        worker_table.setStyle(
            TableStyle(
                [

                    ("GRID", (0, 0), (-1, -1), 1, colors.black),

                    ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),

                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),

                ]
            )
        )

        story.append(worker_table)

        story.append(Spacer(1, 15))

        # ---------------------------------------------------
        # Missing PPE
        # ---------------------------------------------------

        story.append(
            Paragraph(
                "<b>Missing PPE</b>",
                styles["Heading2"],
            )
        )

        if image["missing"]:

            for item in image["missing"]:

                story.append(
                    Paragraph(
                        f"• {item}",
                        styles["Normal"],
                    )
                )

        else:

            story.append(
                Paragraph(
                    "No PPE violations detected.",
                    styles["Normal"],
                )
            )

        story.append(Spacer(1, 15))

        # ---------------------------------------------------
        # Worker Details
        # ---------------------------------------------------

        story.append(
            Paragraph(
                "<b>Worker Details</b>",
                styles["Heading2"],
            )
        )

        for worker in image["workers"]:

            story.append(
                Paragraph(
                    f"""
                    Worker {worker['worker_id']}

                    Status : {worker['status']}

                    Missing PPE :
                    {', '.join(worker['missing']) if worker['missing'] else 'None'}
                    """,
                    styles["Normal"],
                )
            )

            story.append(Spacer(1, 8))

        # ---------------------------------------------------
        # Recommendations
        # ---------------------------------------------------

        story.append(
            Paragraph(
                "<b>AI Recommendations</b>",
                styles["Heading2"],
            )
        )

        if image["status"] == "Unsafe":

            story.append(
                Paragraph(
                    """
                    • Ensure all workers wear required PPE.

                    • Conduct PPE inspections before work begins.

                    • Replace damaged safety equipment immediately.

                    • Provide toolbox safety talks.

                    • Monitor compliance continuously.
                    """,
                    styles["Normal"],
                )
            )

        else:

            story.append(
                Paragraph(
                    "No corrective action required.",
                    styles["Normal"],
                )
            )

        story.append(Spacer(1, 25))

    # ---------------------------------------------------
    # Footer
    # ---------------------------------------------------

    story.append(
        Paragraph(
            "<b>Generated by SAFESIGHT AI</b>",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            "AI-Based Workplace Safety Monitoring System",
            styles["Normal"],
        )
    )

    doc.build(story)