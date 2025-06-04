#!/bin/bash
LAKEHOUSE_APP_NAME=genai-email-demo
APP_FOLDER_IN_WORKSPACE=/Workspace/Users/eric.peter@databricks.com/genai-email-demo

databricks workspace import-dir databricks-app "$APP_FOLDER_IN_WORKSPACE" --overwrite
databricks apps deploy "$LAKEHOUSE_APP_NAME" --source-code-path="$APP_FOLDER_IN_WORKSPACE"
