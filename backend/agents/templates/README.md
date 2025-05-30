# Template Library for Report Generation  
`backend/agents/templates/`

The **Template Engine System** allows any CrewAI agent — primarily the `report_writer` — to transform structured data returned by previous tasks into human-readable artefacts (Markdown, HTML, JSON, plain‐text …).  
Templates are rendered by **TemplateEngineTool** which is a thin wrapper around **Jinja2** with a small set of custom filters and globals that make report creation concise and safe.

---

## 1. Directory layout

```
backend/agents/templates/
├── README.md                 ← this file
├── markdown_report.j2        ← default Markdown template
├── html_report.j2            ← default HTML template
├── json_report.j2            ← default JSON structure
└── <your-template>.j2        ← add more here
```

* Every file **MUST** use the `.j2` suffix.  
* Sub-folders are supported (`fraud/summary.j2`, `alerts/enrichment.j2`, …).

---

## 2. Rendering workflow

1. **Data collection** – upstream agents gather facts (`findings`, `analysis`, `data` etc.).  
2. **TemplateEngineTool** receives:
   ```json
   {
     "template_name": "markdown_report",
     "template_format": "markdown",
     "data": { ... }
   }
   ```
   or dynamic `template_content` as a raw string.
3. The tool loads the template, injects the `data` dict and returns the rendered string to the calling agent.

---

## 3. Available variables

When you call the tool you may supply any keys you like inside `data`.  
In addition, **every** template automatically gets these globals:

| Name             | Type                     | Description                                   |
|------------------|--------------------------|-----------------------------------------------|
| `now()`          | function → `datetime`    | Current date-time (aware)                     |
| `today()`        | function → `date`        | Current date                                   |
| `metadata`       | dict                     | Added automatically if `include_metadata` flag is set |
| `template_format`| str                      | Output format requested (`markdown`, `html` …)|

---

## 4. Built-in filters

| Filter                | Example                                       | Output                               |
|-----------------------|-----------------------------------------------|--------------------------------------|
| `format_date`         | `{{ my_date|format_date('%d %b %Y') }}`       | `30 May 2025`                        |
| `format_currency`     | `{{ amount|format_currency('$',2) }}`         | `$12,345.67`                         |
| `format_percent`      | `{{ ratio|format_percent(1) }}`               | `7.4%`                               |
| `to_json`             | `{{ obj|to_json }}`                           | JSON string                          |
| `truncate_text`       | `{{ long_text|truncate_text(120) }}`          | shortened text with ellipsis         |

You can extend the environment by editing `TemplateEngineTool` (`_initialize_jinja_env`).

---

## 5. Writing a template – quick start

### 5.1 Simple Markdown report

```jinja2
# {{ title }}

**Generated:** {{ now().strftime('%Y-%m-%d %H:%M:%S') }}

## Executive Summary
{{ summary }}

{% if findings %}
## Findings
{% for f in findings %}
* **{{ f.title }}** – {{ f.description }} (Risk: {{ f.risk_level|default('N/A') }})
{% endfor %}
{% endif %}
```

Save as `markdown_report.j2` and call the tool with:

```json
{
  "template_name": "markdown_report",
  "template_format": "markdown",
  "data": {
    "title": "May 2025 Fraud Case",
    "summary": "Multiple high-risk transactions detected …",
    "findings": [
      {"title":"Circular path","description":"$1M looped …","risk_level":"High"}
    ]
  }
}
```

### 5.2 Using a dynamic template

Pass arbitrary content in `template_content` and omit `template_name`.

---

## 6. Adding a new template

1. Create `<name>.j2` under this folder (or a sub-folder).  
2. Reference it via `template_name` **without** the `.j2` suffix.  
3. Commit the file so it is packaged with the backend image.

---

## 7. Security & best practices

* Only Jinja2 rendering is used — `eval`/`exec` are **never** invoked.  
* Avoid user-supplied template strings unless absolutely needed.  
* Large binary artefacts (plots, images) should be generated in the Sandbox and attached separately; include only links/filenames in the report.

---

## 8. Troubleshooting

| Symptom                              | Possible cause / fix                                           |
|--------------------------------------|----------------------------------------------------------------|
| _“TemplateNotFound”_                 | Wrong `template_name`, missing `.j2` file or wrong path.       |
| _“undefined filter”_                 | Custom filter not registered – add it in `TemplateEngineTool`. |
| Invalid JSON output (`json` format)  | Your template emitted non-JSON data – wrap strings in quotes, use `to_json`. |

---

**Happy reporting!**  
Template maintainers ↗ Feel free to extend this README with your own style guide, advanced macros or component libraries.  
Pull-requests are welcome.
