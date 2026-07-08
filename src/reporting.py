import os
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

# ReportLab imports for generating professional PDFs
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to draw running headers, footers, and dynamic page counts
    ('Page X of Y') on all pages except the first page if desired.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Color definitions matching the palette
        slate_color = colors.HexColor("#475569")
        divider_color = colors.HexColor("#e2e8f0")
        
        # Header (Only on pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0f172a"))
            self.drawString(54, 750, "RETAIL SALES INTELLIGENCE & CORRELATION ANALYZER")
            
            self.setFont("Helvetica", 8)
            self.setFillColor(slate_color)
            self.drawRightString(612 - 54, 750, "EXECUTIVE INTELLIGENCE REPORT")
            
            # Divider line
            self.setStrokeColor(divider_color)
            self.setLineWidth(0.75)
            self.line(54, 742, 612 - 54, 742)
        
        # Footer (On all pages)
        self.setStrokeColor(divider_color)
        self.setLineWidth(0.75)
        self.line(54, 52, 612 - 54, 52)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(slate_color)
        self.drawString(54, 38, "CONFIDENTIAL - Retail Sales Intelligence Portfolio")
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 38, page_text)
        
        self.restoreState()


def md_to_html_bold(text: str) -> str:
    """Converts markdown **bold** syntax to ReportLab HTML <b>bold</b> and maps newlines."""
    parts = text.split('**')
    res = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            res.append(p)
        else:
            res.append(f"<b>{p}</b>")
    return "".join(res).replace('\n', '<br/>')


def save_correlation_csv(corr_matrix: pd.DataFrame, output_path: str) -> bool:
    """Saves the correlation matrix as a CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        corr_matrix.to_csv(output_path)
        return True
    except Exception as e:
        print(f"Error saving correlation CSV: {str(e)}")
        return False


def generate_text_report(
    dataset_name: str,
    shape: tuple,
    cleaning_summary: Dict[str, Any],
    pos_corrs: pd.DataFrame,
    neg_corrs: pd.DataFrame,
    insights: List[Dict[str, Any]],
    disclaimer: Dict[str, str],
    dataset_insights: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Generates a structured, professional plaintext report.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("        RETAIL SALES INTELLIGENCE & CORRELATION EXECUTIVE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated On: {timestamp}\n")
            f.write(f"Dataset Analyzed: {dataset_name}\n")
            f.write(f"Dimensions: {shape[0]} rows, {shape[1]} columns\n")
            f.write("-" * 80 + "\n\n")
            
            # Data Preprocessing Summary
            f.write("1. DATA PREPROCESSING SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"- Duplicates Removed: {cleaning_summary.get('duplicates_removed', 0):,}\n")
            f.write(f"- Missing Customer IDs Removed: {cleaning_summary.get('null_customers_removed', 0):,}\n")
            f.write(f"- Missing Descriptions Removed: {cleaning_summary.get('null_descriptions_removed', 0):,}\n")
            f.write(f"- Returned Transactions Separated: {cleaning_summary.get('returns_count', 0):,} lines\n")
            f.write(f"  * Total Returned Items: {cleaning_summary.get('returns_items', 0):,}\n")
            f.write(f"  * Est Returned Value: ${cleaning_summary.get('returns_value', 0.0):,.2f}\n")
            f.write(f"- Zero/Negative Prices Filtered: {cleaning_summary.get('invalid_prices_removed', 0):,}\n")
            f.write(f"- Date Columns Parsed: {', '.join(cleaning_summary.get('date_conversions', [])) or 'None'}\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Top Correlations
            f.write("2. CORRELATION RANKINGS\n")
            f.write("-" * 25 + "\n")
            f.write("Top Strongest Positive Correlations:\n")
            if not pos_corrs.empty:
                for idx, row in pos_corrs.iterrows():
                    f.write(f"  {idx+1}. {row['Variable 1']} <--> {row['Variable 2']}: {row['Correlation']:.4f}\n")
            else:
                f.write("  No positive correlations found.\n")
                
            f.write("\nTop Strongest Negative Correlations:\n")
            if not neg_corrs.empty:
                for idx, row in neg_corrs.iterrows():
                    f.write(f"  {idx+1}. {row['Variable 1']} <--> {row['Variable 2']}: {row['Correlation']:.4f}\n")
            else:
                f.write("  No negative correlations found.\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Highlighted Dataset Business Insights
            f.write("3. HIGHLIGHTED DATASET BUSINESS INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"- Top-Performing Country by Revenue: {dataset_insights.get('top_country', 'N/A')} (Revenue: ${dataset_insights.get('top_country_revenue', 0.0):,.2f})\n")
            f.write(f"- Best-Selling Product: {dataset_insights.get('best_selling_product', 'N/A')} ({dataset_insights.get('best_selling_units', 0):,} units sold)\n")
            f.write(f"- Highest Revenue Month: {dataset_insights.get('highest_rev_month', 'N/A')} (Revenue: ${dataset_insights.get('highest_rev_month_revenue', 0.0):,.2f})\n")
            f.write(f"- Average Order Value (AOV): ${dataset_insights.get('aov', 0.0):,.2f}\n\n")
            f.write("Notable Trends & Anomalies:\n")
            for trend in dataset_insights.get('trends', []):
                f.write(f"  * {trend}\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Business Insights
            f.write("4. CORRELATION BUSINESS INSIGHTS & EXPLANATIONS\n")
            f.write("-" * 45 + "\n")
            for idx, insight in enumerate(insights):
                f.write(f"Insight #{idx+1} [{insight['type']} Correlation, Strength: {insight['strength']}]\n")
                f.write(f"Relationship: {insight['var1']} and {insight['var2']} ({insight['correlation']:.2f})\n")
                f.write(f"Interpretation: {insight['text']}\n\n")
            f.write("-" * 80 + "\n\n")
            
            # Causal Disclaimer
            f.write(f"5. CRITICAL SCIENTIFIC WARNING: {disclaimer['headline'].upper()}\n")
            f.write("-" * 65 + "\n")
            f.write(disclaimer['core_concept'] + "\n\n")
            f.write("Key Factors to Note:\n")
            f.write(disclaimer['retail_reasons'] + "\n\n")
            f.write("Recommendation:\n")
            f.write(disclaimer['actionable_advice'] + "\n")
            f.write("=" * 80 + "\n")
            
        return True
    except Exception as e:
        print(f"Error generating text report: {str(e)}")
        return False


def generate_pdf_report(
    dataset_name: str,
    shape: tuple,
    cleaning_summary: Dict[str, Any],
    pos_corrs: pd.DataFrame,
    neg_corrs: pd.DataFrame,
    insights: List[Dict[str, Any]],
    disclaimer: Dict[str, str],
    dataset_insights: Dict[str, Any],
    heatmap_path: Optional[str],
    scatter_path: Optional[str],
    output_path: str
) -> bool:
    """
    Generates a professional, print-ready PDF executive report using ReportLab.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup document template with customized margins to clear header/footer lines
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=80,
            bottomMargin=80
        )
        
        # Load styles and set custom styling
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=15
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=25
        )
        
        heading1_style = ParagraphStyle(
            'Heading1_Custom',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e3a8a"),  # Deep blue primary
            spaceBefore=15,
            spaceAfter=10,
            keepWithNext=True
        )

        heading2_style = ParagraphStyle(
            'Heading2_Custom',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'Body_Custom',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8
        )
        
        meta_style = ParagraphStyle(
            'Meta_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1e293b")
        )
        
        insight_title_style = ParagraphStyle(
            'InsightTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#0f172a")
        )
        
        warning_style = ParagraphStyle(
            'WarningText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#7f1d1d")
        )
        
        warning_bold_style = ParagraphStyle(
            'WarningBoldText',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#7f1d1d")
        )
        
        story = []
        
        # --- TITLE BLOCK ---
        story.append(Spacer(1, 10))
        story.append(Paragraph("RETAIL BUSINESS INTELLIGENCE SYSTEM REPORT", title_style))
        story.append(Paragraph(
            f"<b>Analyzed Dataset:</b> {dataset_name} | <b>Date Generated:</b> {datetime.date.today().strftime('%B %d, %Y')}",
            subtitle_style
        ))
        
        # --- DATASET OVERVIEW (Table Grid) ---
        story.append(Paragraph("1. Data Preprocessing & Scope", heading1_style))
        story.append(Paragraph(
            "Below is a summary of the dataset scope and the data cleaning rules executed during ingestion:",
            body_style
        ))
        
        overview_data = [
            [Paragraph("<b>Metric</b>", meta_style), Paragraph("<b>Value / Processing Applied</b>", meta_style)],
            [Paragraph("Active Sales Records (Rows)", meta_style), Paragraph(f"{shape[0]:,}", meta_style)],
            [Paragraph("Engineered Features (Columns)", meta_style), Paragraph(f"{shape[1]}", meta_style)],
            [Paragraph("Duplicates Removed", meta_style), Paragraph(f"{cleaning_summary.get('duplicates_removed', 0):,} rows", meta_style)],
            [Paragraph("Missing Customer IDs Removed", meta_style), Paragraph(f"{cleaning_summary.get('null_customers_removed', 0):,} rows", meta_style)],
            [Paragraph("Missing Descriptions Removed", meta_style), Paragraph(f"{cleaning_summary.get('null_descriptions_removed', 0):,} rows", meta_style)],
            [Paragraph("Returned Orders Separated", meta_style), Paragraph(f"{cleaning_summary.get('returns_count', 0):,} items (Qty: {cleaning_summary.get('returns_items', 0):,}, Val: ${cleaning_summary.get('returns_value', 0.0):,.2f})", meta_style)],
            [Paragraph("Invalid Prices Filtered", meta_style), Paragraph(f"{cleaning_summary.get('invalid_prices_removed', 0):,} rows", meta_style)],
        ]
            
        overview_table = Table(overview_data, colWidths=[2.0*inch, 4.5*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (1,0), colors.HexColor("#0f172a")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 15))
        
        # --- CORRELATION RANKINGS ---
        story.append(Paragraph("2. Pearson Correlation Rankings", heading1_style))
        story.append(Paragraph(
            "The top relationships ranked by Pearson correlation coefficient strength (r) in both positive and negative directions:",
            body_style
        ))
        
        rankings_data = [
            [Paragraph("<b>Rank</b>", meta_style), Paragraph("<b>Variable A</b>", meta_style), Paragraph("<b>Variable B</b>", meta_style), Paragraph("<b>Coefficient (r)</b>", meta_style), Paragraph("<b>Direction</b>", meta_style)]
        ]
        
        # Add top positive
        pos_added = 0
        for _, row in pos_corrs.head(3).iterrows():
            pos_added += 1
            rankings_data.append([
                Paragraph(f"Pos #{pos_added}", meta_style),
                Paragraph(row['Variable 1'], meta_style),
                Paragraph(row['Variable 2'], meta_style),
                Paragraph(f"<b>{row['Correlation']:.4f}</b>", meta_style),
                Paragraph("<font color='#16a34a'>Positive</font>", meta_style)
            ])
            
        # Add top negative
        neg_added = 0
        for _, row in neg_corrs.head(3).iterrows():
            neg_added += 1
            rankings_data.append([
                Paragraph(f"Neg #{neg_added}", meta_style),
                Paragraph(row['Variable 1'], meta_style),
                Paragraph(row['Variable 2'], meta_style),
                Paragraph(f"<b>{row['Correlation']:.4f}</b>", meta_style),
                Paragraph("<font color='#dc2626'>Negative</font>", meta_style)
            ])
            
        rank_table = Table(rankings_data, colWidths=[0.8*inch, 2.0*inch, 2.0*inch, 1.0*inch, 1.2*inch])
        rank_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (4,0), colors.HexColor("#f1f5f9")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(rank_table)
        
        # Force a page break for insights and visual layouts to keep it neat
        story.append(PageBreak())
        
        # --- EXECUTIVE BUSINESS & DATASET INSIGHTS ---
        story.append(Paragraph("3. Executive Business & Dataset Insights", heading1_style))
        story.append(Paragraph(
            "High-level operational metrics and key performance indicators generated from the active filtered dataset:",
            body_style
        ))
        
        kpi_data = [
            [Paragraph("<b>Key KPI Variable</b>", meta_style), Paragraph("<b>Performance Interpretation</b>", meta_style)],
            [Paragraph("Top Country by Revenue", meta_style), Paragraph(f"<b>{dataset_insights.get('top_country', 'N/A')}</b> (Revenue: ${dataset_insights.get('top_country_revenue', 0.0):,.2f})", meta_style)],
            [Paragraph("Best-Selling Product", meta_style), Paragraph(f"<b>{dataset_insights.get('best_selling_product', 'N/A')}</b> ({dataset_insights.get('best_selling_units', 0):,} units sold)", meta_style)],
            [Paragraph("Highest Revenue Month", meta_style), Paragraph(f"<b>{dataset_insights.get('highest_rev_month', 'N/A')}</b> (Revenue: ${dataset_insights.get('highest_rev_month_revenue', 0.0):,.2f})", meta_style)],
            [Paragraph("Average Order Value (AOV)", meta_style), Paragraph(f"${dataset_insights.get('aov', 0.0):,.2f}", meta_style)],
        ]
        kpi_table = Table(kpi_data, colWidths=[2.2*inch, 4.3*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor("#f1f5f9")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Notable Trends & Operational Anomalies:</b>", heading2_style))
        for trend in dataset_insights.get('trends', []):
            story.append(Paragraph(f"• {trend}", body_style))
        story.append(Spacer(1, 15))
        
        # --- CORRELATION BUSINESS INSIGHTS ---
        story.append(Paragraph("4. Correlation Business Insights", heading1_style))
        story.append(Paragraph(
            "Practical operational interpretations of the observed correlation variables:",
            body_style
        ))
        
        # Display up to 4 top insights
        for idx, insight in enumerate(insights[:4]):
            badge_color = "#16a34a" if insight['type'] == 'Positive' else "#dc2626"
            header_text = f"<b>Insight #{idx+1}: {insight['var1']} & {insight['var2']}</b> (r = {insight['correlation']:.2f}) | <font color='{badge_color}'><b>{insight['type']}</b></font> ({insight['strength']})"
            
            insight_story = [
                Paragraph(header_text, insight_title_style),
                Spacer(1, 3),
                Paragraph(insight['text'], body_style),
                Spacer(1, 8)
            ]
            story.append(KeepTogether(insight_story))
            
        story.append(Spacer(1, 10))
        
        # --- CAUSATION DISCLAIMER BLOCK (Callout Box) ---
        disclaimer_data = [
            [
                Paragraph(f"<b>CRITICAL SCIENTIFIC CAVEAT: {disclaimer['headline'].upper()}</b>", warning_bold_style)
            ],
            [
                Paragraph(disclaimer['core_concept'], warning_style)
            ],
            [
                Paragraph(f"<b>Causal Fallacy Risks in Retail:</b><br/>{md_to_html_bold(disclaimer['retail_reasons'])}", warning_style)
            ],
            [
                Paragraph(f"<b>Actionable Directive:</b> {disclaimer['actionable_advice']}", warning_style)
            ]
        ]
        
        disclaimer_table = Table(disclaimer_data, colWidths=[6.5*inch])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2")),  # Light reddish callout
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fca5a5")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        
        story.append(KeepTogether([disclaimer_table]))
        
        # --- VISUALIZATIONS PAGE (Page 3) ---
        if heatmap_path or scatter_path:
            story.append(PageBreak())
            story.append(Paragraph("5. Data Visualization Appendix", heading1_style))
            story.append(Paragraph(
                "Visual representation of the correlation coefficients and selected retail relationships:",
                body_style
            ))
            
            # Embed heatmap
            if heatmap_path and os.path.exists(heatmap_path):
                img_w = 4.5 * inch
                img_h = 3.6 * inch  # maintains 10:8 aspect ratio
                h_img = Image(heatmap_path, width=img_w, height=img_h)
                story.append(Paragraph("<b>Figure A: Upper-Triangle Masked Correlation Heatmap</b>", heading2_style))
                story.append(KeepTogether([h_img, Spacer(1, 15)]))
                
            # Embed scatter plot
            if scatter_path and os.path.exists(scatter_path):
                img_w = 4.5 * inch
                img_h = 3.375 * inch  # maintains 8:6 aspect ratio
                s_img = Image(scatter_path, width=img_w, height=img_h)
                story.append(Paragraph("<b>Figure B: Key Retail Relationship Scatter Analysis</b>", heading2_style))
                story.append(KeepTogether([s_img]))
                
        # Build Document using NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)
        return True
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        # Import traceback for thorough error capturing in logs
        import traceback
        traceback.print_exc()
        return False
