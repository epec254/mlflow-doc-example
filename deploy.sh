#!/bin/bash
LAKEHOUSE_APP_NAME=genai-email-demo
APP_FOLDER_IN_WORKSPACE=/Workspace/Users/eric.peter@databricks.com/genai-email-demo

# Function to update app.yaml with environment variables from .env
update_app_yaml() {
    local env_file=".env"
    local app_yaml="databricks-app/app.yaml"
    
    # Check if .env file exists
    if [[ ! -f "$env_file" ]]; then
        echo "Warning: .env file not found. Using existing app.yaml without modifications."
        return
    fi
    
    echo "Reading environment variables from $env_file..."
    
    # Create temporary files to store variables
    local temp_vars=$(mktemp)
    local temp_yaml=$(mktemp)
    
    # First pass: clean and store all variables
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip empty lines and comments
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        
        # Remove any surrounding quotes and whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        
        # Remove quotes if they exist
        if [[ "$value" =~ ^[\"\'].*[\"\']$ ]]; then
            value="${value:1:-1}"
        fi
        
        # Store in temporary file
        echo "$key=$value" >> "$temp_vars"
        
    done < "$env_file"
    
    # Function to resolve variable references
    resolve_var_refs() {
        local value="$1"
        local resolved="$value"
        
        # Replace ${VAR_NAME} with actual values from temp_vars file
        while [[ "$resolved" =~ \$\{([^}]+)\} ]]; do
            local var_name="${BASH_REMATCH[1]}"
            local var_value=""
            
            # Look up the variable in our temp file
            if grep -q "^$var_name=" "$temp_vars"; then
                var_value=$(grep "^$var_name=" "$temp_vars" | head -1 | cut -d'=' -f2-)
            fi
            
            if [[ -n "$var_value" ]]; then
                resolved="${resolved/\$\{$var_name\}/$var_value}"
            else
                echo "Warning: Variable reference \${$var_name} not found in .env file"
                break
            fi
        done
        echo "$resolved"
    }
    
    # Write the command section
    cat > "$temp_yaml" << 'EOF'
command: [
  "uvicorn",
  "app:app"
]

env:
EOF
    
    # Second pass: process variables and write to YAML
    while IFS='=' read -r key value; do
        [[ -z "$key" ]] && continue
        
        # Skip Databricks authentication variables (used for deployment, not runtime)
        if [[ "$key" == "DATABRICKS_TOKEN" || "$key" == "DATABRICKS_HOST" ]]; then
            echo "Skipping $key (deployment-only variable)"
            continue
        fi
        
        # Resolve any variable references in the value
        local resolved_value
        resolved_value=$(resolve_var_refs "$value")
        
        # Add to YAML format
        echo "  - name: '$key'" >> "$temp_yaml"
        echo "    value: '$resolved_value'" >> "$temp_yaml"
        
    done < "$temp_vars"
    
    # Replace the original app.yaml
    mv "$temp_yaml" "$app_yaml"
    rm -f "$temp_vars"
    echo "Updated $app_yaml with environment variables from $env_file"
}

# Update app.yaml with environment variables
update_app_yaml

# Deploy the application
databricks workspace import-dir databricks-app "$APP_FOLDER_IN_WORKSPACE" --overwrite
databricks apps deploy "$LAKEHOUSE_APP_NAME" --source-code-path="$APP_FOLDER_IN_WORKSPACE"
