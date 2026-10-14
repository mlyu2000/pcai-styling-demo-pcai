{{- define "pcai.styling.name" -}}
styling-demo
{{- end -}}
{{- define "pcai.styling.backend" -}}
styling-backend
{{- end -}}
{{- define "pcai.styling.frontend" -}}
styling-frontend
{{- end -}}
{{- define "pcai.styling.labels" -}}
app.kubernetes.io/part-of: pcai-styling-demo
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
