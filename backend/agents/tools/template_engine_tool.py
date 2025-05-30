"""
TemplateEngineTool for generating reports from templates.

This tool provides CrewAI agents with the ability to generate reports from
Jinja2-style templates. It supports various output formats including Markdown,
HTML, and structured JSON for visualization tools. The tool is primarily used
by the report_writer agent to create consistent, well-formatted reports.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import datetime
import re

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default directory for report templates
TEMPLATE_DIR = Path("backend/agents/templates")


class TemplateInput(BaseModel):
    """Input model for template engine."""
    
    template_name: Optional[str] = Field(
        default=None,
        description="Name of the template to use (if using a predefined template)"
    )
    template_content: Optional[str] = Field(
        default=None,
        description="Raw template content (if not using a predefined template)"
    )
    template_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'html', 'json', or 'text'"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Data to render in the template"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in the output (timestamp, etc.)"
    )


class TemplateEngineTool(BaseTool):
    """
    Tool for generating reports from templates using Jinja2.
    
    This tool allows agents to generate formatted reports using Jinja2 templates.
    It supports various output formats and can use either predefined templates
    or dynamically provided template content.
    """
    
    name: str = "template_engine_tool"
    description: str = """
    Generate reports from templates using Jinja2.
    
    Use this tool when you need to:
    - Create formatted reports from data
    - Generate consistent documentation
    - Produce executive summaries
    - Format analysis results for presentation
    - Create structured outputs for downstream processing
    
    The tool supports various output formats including Markdown, HTML, and JSON.
    You can use either predefined templates or provide custom template content.
    
    Example usage:
    - Generate a fraud investigation report from analysis results
    - Create an executive summary of suspicious activity
    - Format transaction data for presentation
    - Generate a risk assessment report
    """
    args_schema: type[BaseModel] = TemplateInput
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the TemplateEngineTool.
        
        Args:
            template_dir: Optional custom directory for templates
        """
        super().__init__()
        
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available. Install with 'pip install jinja2'")
        
        self.template_dir = template_dir or TEMPLATE_DIR
        self.env = self._initialize_jinja_env() if JINJA2_AVAILABLE else None
        self.templates = self._load_templates() if JINJA2_AVAILABLE else {}
    
    def _initialize_jinja_env(self) -> 'jinja2.Environment':
        """
        Initialize the Jinja2 environment with custom filters and functions.
        
        Returns:
            Configured Jinja2 Environment
        """
        # Create template loader
        os.makedirs(self.template_dir, exist_ok=True)
        loader = jinja2.FileSystemLoader(self.template_dir)
        
        # Create environment
        env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        env.filters['format_date'] = lambda d, fmt='%Y-%m-%d': d.strftime(fmt) if isinstance(d, datetime.datetime) else d
        env.filters['format_currency'] = lambda v, symbol='$', decimals=2: f"{symbol}{v:,.{decimals}f}" if v is not None else ''
        env.filters['format_percent'] = lambda v, decimals=2: f"{v:.{decimals}f}%" if v is not None else ''
        env.filters['to_json'] = lambda v: json.dumps(v, default=str)
        env.filters['truncate_text'] = lambda text, length=100: text[:length] + '...' if len(text) > length else text
        
        # Add custom global functions
        env.globals['now'] = datetime.datetime.now
        env.globals['today'] = datetime.date.today
        
        return env
    
    def _load_templates(self) -> Dict[str, Any]:
        """
        Load all templates from the template directory.
        
        Returns:
            Dictionary of template instances by name
        """
        templates = {}
        
        # Load all template files
        for file_path in self.template_dir.glob("**/*.j2"):
            template_name = file_path.stem
            try:
                templates[template_name] = self.env.get_template(file_path.name)
                logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_name}: {e}")
        
        # If no templates were found, create default templates
        if not templates:
            logger.info("No templates found, creating default templates")
            self._create_default_templates()
            
            # Load the newly created templates
            for file_path in self.template_dir.glob("**/*.j2"):
                template_name = file_path.stem
                try:
                    templates[template_name] = self.env.get_template(file_path.name)
                    logger.debug(f"Loaded template: {template_name}")
                except Exception as e:
                    logger.error(f"Error loading template {template_name}: {e}")
        
        logger.info(f"Loaded {len(templates)} templates")
        return templates
    
    def _create_default_templates(self):
        """Create default templates if none exist."""
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Markdown report template
        markdown_template = """# {{ title }}

**Generated:** {{ now().strftime('%Y-%m-%d %H:%M:%S') }}
{% if author %}**Author:** {{ author }}{% endif %}

## Executive Summary

{{ summary }}

## Key Findings

{% for finding in findings %}
### {{ finding.title }}

{{ finding.description }}

{% if finding.risk_level %}**Risk Level:** {{ finding.risk_level }}{% endif %}
{% if finding.evidence %}**Evidence:** {{ finding.evidence }}{% endif %}

{% endfor %}

## Detailed Analysis

{{ analysis }}

{% if recommendations %}
## Recommendations

{% for rec in recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

{% if data %}
## Data Summary

| Metric | Value |
|--------|-------|
{% for key, value in data.items() %}
| {{ key }} | {{ value }} |
{% endfor %}
{% endif %}

{% if metadata %}
---
**Metadata:**
{% for key, value in metadata.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}
"""
        
        # HTML report template
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        h2 { color: #3498db; margin-top: 30px; }
        h3 { color: #2980b9; }
        .executive-summary { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }
        .finding { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
        .finding h3 { margin-top: 0; }
        .high-risk { border-left: 4px solid #e74c3c; }
        .medium-risk { border-left: 4px solid #f39c12; }
        .low-risk { border-left: 4px solid #2ecc71; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .metadata { font-size: 0.8em; color: #7f8c8d; border-top: 1px solid #eee; margin-top: 40px; padding-top: 10px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p><strong>Generated:</strong> {{ now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
    {% if author %}<p><strong>Author:</strong> {{ author }}</p>{% endif %}
    
    <h2>Executive Summary</h2>
    <div class="executive-summary">
        {{ summary }}
    </div>
    
    <h2>Key Findings</h2>
    {% for finding in findings %}
    <div class="finding {% if finding.risk_level %}{{ finding.risk_level|lower }}-risk{% endif %}">
        <h3>{{ finding.title }}</h3>
        <p>{{ finding.description }}</p>
        {% if finding.risk_level %}<p><strong>Risk Level:</strong> {{ finding.risk_level }}</p>{% endif %}
        {% if finding.evidence %}<p><strong>Evidence:</strong> {{ finding.evidence }}</p>{% endif %}
    </div>
    {% endfor %}
    
    <h2>Detailed Analysis</h2>
    <div>
        {{ analysis|safe }}
    </div>
    
    {% if recommendations %}
    <h2>Recommendations</h2>
    <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
    {% endif %}
    
    {% if data %}
    <h2>Data Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        {% for key, value in data.items() %}
        <tr>
            <td>{{ key }}</td>
            <td>{{ value }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    
    {% if metadata %}
    <div class="metadata">
        <p><strong>Metadata:</strong></p>
        <ul>
            {% for key, value in metadata.items() %}
            <li>{{ key }}: {{ value }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
"""
        
        # JSON report template (simple structure)
        json_template = """{
    "report": {
        "title": "{{ title }}",
        "generated_at": "{{ now().isoformat() }}",
        {% if author %}"author": "{{ author }}",{% endif %}
        "summary": "{{ summary }}",
        "findings": [
            {% for finding in findings %}
            {
                "title": "{{ finding.title }}",
                "description": "{{ finding.description }}",
                {% if finding.risk_level %}"risk_level": "{{ finding.risk_level }}",{% endif %}
                {% if finding.evidence %}"evidence": "{{ finding.evidence }}",{% endif %}
                {% if finding.score %}"score": {{ finding.score }},{% endif %}
                {% if finding.metrics %}"metrics": {{ finding.metrics|to_json }},{% endif %}
                "id": "{{ loop.index }}"
            }{% if not loop.last %},{% endif %}
            {% endfor %}
        ],
        "analysis": "{{ analysis|replace('\n', ' ')|replace('\"', '\\\"') }}",
        {% if recommendations %}
        "recommendations": [
            {% for rec in recommendations %}
            "{{ rec }}"{% if not loop.last %},{% endif %}
            {% endfor %}
        ],
        {% endif %}
        {% if data %}
        "data": {{ data|to_json }},
        {% endif %}
        {% if metadata %}
        "metadata": {{ metadata|to_json }}
        {% endif %}
    }
}
"""
        
        # Save templates to files
        with open(self.template_dir / "markdown_report.j2", "w") as f:
            f.write(markdown_template)
        
        with open(self.template_dir / "html_report.j2", "w") as f:
            f.write(html_template)
        
        with open(self.template_dir / "json_report.j2", "w") as f:
            f.write(json_template)
        
        logger.info("Created default templates")
    
    async def _arun(
        self,
        template_name: Optional[str] = None,
        template_content: Optional[str] = None,
        template_format: str = "markdown",
        data: Dict[str, Any] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Generate a report from a template asynchronously.
        
        Args:
            template_name: Name of the template to use
            template_content: Raw template content
            template_format: Output format
            data: Data to render in the template
            include_metadata: Whether to include metadata
            
        Returns:
            JSON string containing the rendered report
        """
        try:
            if not JINJA2_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "Jinja2 is not available. Install with 'pip install jinja2'"
                })
            
            # Ensure data is provided
            data = data or {}
            
            # Add metadata if requested
            if include_metadata:
                data["metadata"] = {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "tool": "TemplateEngineTool",
                    "template_format": template_format
                }
            
            # Determine which template to use
            template = None
            
            if template_content:
                # Use provided template content
                template = self.env.from_string(template_content)
                template_source = "custom"
            elif template_name:
                # Use named template
                if template_name in self.templates:
                    template = self.templates[template_name]
                    template_source = f"named:{template_name}"
                else:
                    # Try to load by filename
                    try:
                        template = self.env.get_template(f"{template_name}.j2")
                        template_source = f"file:{template_name}.j2"
                    except jinja2.exceptions.TemplateNotFound:
                        template = None
            
            # If no template found, use default based on format
            if not template:
                default_templates = {
                    "markdown": "markdown_report.j2",
                    "html": "html_report.j2",
                    "json": "json_report.j2",
                    "text": "markdown_report.j2"  # Fallback to markdown for text
                }
                
                template_file = default_templates.get(template_format.lower(), "markdown_report.j2")
                try:
                    template = self.env.get_template(template_file)
                    template_source = f"default:{template_file}"
                except jinja2.exceptions.TemplateNotFound:
                    # Create default templates and try again
                    self._create_default_templates()
                    template = self.env.get_template(template_file)
                    template_source = f"default:{template_file}"
            
            # Render the template
            rendered_content = template.render(**data)
            
            # Clean up the output based on format
            if template_format.lower() == "json":
                # Validate JSON
                try:
                    json_content = json.loads(rendered_content)
                    rendered_content = json.dumps(json_content, indent=2)
                except json.JSONDecodeError as e:
                    logger.warning(f"Generated JSON is invalid: {e}")
                    # Return as-is, don't try to fix
            elif template_format.lower() == "markdown":
                # Clean up markdown (remove extra blank lines)
                rendered_content = re.sub(r'\n{3,}', '\n\n', rendered_content)
            
            return json.dumps({
                "success": True,
                "template_source": template_source,
                "template_format": template_format,
                "content": rendered_content
            })
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run(
        self,
        template_name: Optional[str] = None,
        template_content: Optional[str] = None,
        template_format: str = "markdown",
        data: Dict[str, Any] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(template_name, template_content, template_format, data, include_metadata)
        )
