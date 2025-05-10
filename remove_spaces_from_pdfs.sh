#!/bin/bash

# Script to remove spaces from filenames in a specified directory.
# Usage: ./remove_spaces.sh [directory_path]
# If directory_path is omitted, it attempts to read the path from
# src/config.yaml (relative to this script's location).
# If that fails, it defaults to the current directory.
# Warns and overwrites if the new filename already exists.

# --- Configuration --- 
# Get the absolute path of the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CONFIG_FILE="$SCRIPT_DIR/src/config.yaml"
CONFIG_KEY="PDFS_TO_CHAT_WITH_DIRECTORY"

# --- Determine Target Directory --- 
TARGET_DIR=""

if [ -n "$1" ]; then
  # Use provided argument if it exists
  TARGET_DIR="$1"
  echo "Using provided directory: '$TARGET_DIR'"
elif [ -f "$CONFIG_FILE" ]; then
  # Attempt to parse config file if no argument provided and file exists
  echo "No directory provided, attempting to read from '$CONFIG_FILE'..."
  # Use grep to find the line, sed to extract the value (handles optional quotes, strips comments)
  # More robust parsing: find line, remove key, remove quotes, remove comment
  RELATIVE_PDF_DIR=$(grep "^${CONFIG_KEY}:" "$CONFIG_FILE" | sed -e 's/^[^:]*:[ 	]*//' -e 's/"//g' -e 's/[ 	]*#.*//' -e 's/[[:space:]]*$//')
  
  if [ -n "$RELATIVE_PDF_DIR" ]; then
    # Construct absolute path
    # Ensure no double slashes if SCRIPT_DIR is / and RELATIVE_PDF_DIR starts with /
    # Or if RELATIVE_PDF_DIR ends with / and SCRIPT_DIR doesn't.
    # Simplest approach: use realpath for canonicalization if available, or simple concatenation.
    TARGET_DIR="$SCRIPT_DIR/$RELATIVE_PDF_DIR" 
    # Basic normalization: remove trailing slash if it exists for consistency
    TARGET_DIR="${TARGET_DIR%/}" 
    echo "Read relative path '$RELATIVE_PDF_DIR' from config. Using target directory: '$TARGET_DIR'"
  else
    echo "Warning: Could not parse key '$CONFIG_KEY' from '$CONFIG_FILE'. Defaulting to current directory ('.')."
    TARGET_DIR="."
  fi
else
  # Config file not found, default to current directory
  echo "Warning: Config file '$CONFIG_FILE' not found. Defaulting to current directory ('.')."
  TARGET_DIR="."
fi

# Check if the target directory exists and is a directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found or is not a directory."
  exit 1
fi

echo "Processing files in '$TARGET_DIR'..."

# Find files in the target directory (not recursively, only top level)
# Use null delimiter for safe handling of filenames with special characters
find "$TARGET_DIR" -maxdepth 1 -type f -print0 | while IFS= read -r -d $'\0' file; do
  # Get the base filename
  filename=$(basename "$file")
  # Get the directory path
  dirname=$(dirname "$file")

  # Create the new filename by removing all spaces
  new_filename=$(echo "$filename" | tr -d ' ')

  # Proceed only if the name actually changes
  if [ "$filename" != "$new_filename" ]; then
    # Construct the full path for the new filename
    new_filepath="$dirname/$new_filename"

    # Check if the target path exists
    if [ -e "$new_filepath" ]; then
      # Ensure the existing item is a file before overwriting
      if [ -f "$new_filepath" ]; then
          echo "Warning: Renaming '$filename' to '$new_filename'. Overwriting existing file."
          # Perform the rename, forcing overwrite
          mv -f "$file" "$new_filepath"
          if [ $? -eq 0 ]; then
            echo "Successfully renamed '$filename' -> '$new_filename' (overwrote existing)."
          else
            echo "Error: Failed to rename '$filename' to '$new_filename'."
          fi
      else
          # Target exists but is not a regular file (e.g., a directory)
          echo "Error: Cannot rename '$filename' to '$new_filename'. A non-file item with the target name already exists."
      fi
    else
      # Target path does not exist, proceed with normal rename
      mv "$file" "$new_filepath"
      if [ $? -eq 0 ]; then
        echo "Successfully renamed '$filename' -> '$new_filename'."
      else
        echo "Error: Failed to rename '$filename' to '$new_filename'."
      fi
    fi
  fi
done

echo "Finished processing files in '$TARGET_DIR'."
exit 0
