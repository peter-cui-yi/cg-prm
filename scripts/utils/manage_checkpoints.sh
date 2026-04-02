#!/bin/bash
# Manage training checkpoints (backup, prune, export)
# Usage: bash scripts/manage_checkpoints.sh [backup|prune|export] [OPTIONS]

set -e

ACTION="${1:-help}"
OUTPUT_DIR="${2:-outputs}"

echo "=============================================="
echo "Checkpoint Manager"
echo "=============================================="
echo "Action: $ACTION"
echo ""

case "$ACTION" in
  backup)
    BACKUP_DIR="${3:-checkpoints_backup_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$BACKUP_DIR"
    
    echo "Backing up checkpoints from $OUTPUT_DIR..."
    for dir in "$OUTPUT_DIR"/*/; do
      if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        echo "  Backing up $model_name..."
        cp -r "$dir" "$BACKUP_DIR/" 2>/dev/null || true
      fi
    done
    
    echo "Backup complete: $BACKUP_DIR"
    ;;
  
  prune)
    KEEP="${3:-2}"
    
    echo "Pruning checkpoints, keeping top $KEEP..."
    for dir in "$OUTPUT_DIR"/*/; do
      if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        echo "  Pruning $model_name..."
        
        # Get all checkpoint directories sorted by step number
        checkpoints=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V)
        num_checkpoints=$(echo "$checkpoints" | wc -l)
        
        if [ "$num_checkpoints" -gt "$KEEP" ]; then
          to_remove=$((num_checkpoints - KEEP))
          echo "    Removing $to_remove old checkpoints..."
          echo "$checkpoints" | head -n "$to_remove" | xargs rm -rf
        else
          echo "    Only $num_checkpoints checkpoints found, keeping all"
        fi
      fi
    done
    
    echo "Pruning complete"
    ;;
  
  export)
    EXPORT_DIR="${3:-exported_models}"
    mkdir -p "$EXPORT_DIR"
    
    echo "Exporting merged models to $EXPORT_DIR..."
    
    # This would require merging LoRA weights with base model
    # Placeholder for now
    echo "  Export not yet implemented - requires LoRA merge script"
    echo "  Use PEFT's merge_and_unload() to export full models"
    ;;
  
  help|*)
    echo "Usage:"
    echo "  $0 backup [output_dir] [backup_dir]"
    echo "  $0 prune [output_dir] [keep_count]"
    echo "  $0 export [output_dir] [export_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 backup outputs checkpoints_backup_20260401"
    echo "  $0 prune outputs 2"
    echo "  $0 export outputs exported_models"
    ;;
esac

echo ""
