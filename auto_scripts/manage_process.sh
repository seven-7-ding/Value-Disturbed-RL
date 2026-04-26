#!/bin/bash

# ============= Interactive Process Kill Script =============
# This script allows you to search and kill processes interactively

echo "============================================"
echo "Interactive Process Kill Tool"
echo "============================================"
echo ""
echo "Please enter search patterns to filter processes."
echo "You can enter multiple patterns (press Enter after each)."
echo "Type 'done' when you're finished adding patterns."
echo ""

# Array to store search patterns
declare -a PATTERNS=()

# Collect search patterns from user
while true; do
    read -p "Enter search pattern (or 'done' to finish): " pattern
    
    if [ "$pattern" = "done" ]; then
        break
    fi
    
    if [ -n "$pattern" ]; then
        PATTERNS+=("$pattern")
        echo "  ✓ Added pattern: '$pattern'"
    fi
done

echo ""

# Check if any patterns were provided
if [ ${#PATTERNS[@]} -eq 0 ]; then
    echo "❌ No search patterns provided. Exiting."
    exit 0
fi

echo "============================================"
echo "Searching with ${#PATTERNS[@]} pattern(s):"
for pattern in "${PATTERNS[@]}"; do
    echo "  - '$pattern'"
done
echo "============================================"
echo ""

# Build grep command with all patterns
GREP_CMD="ps aux"
for pattern in "${PATTERNS[@]}"; do
    GREP_CMD="$GREP_CMD | grep '$pattern'"
done
GREP_CMD="$GREP_CMD | grep -v grep"

# Execute the search
PROCESS_LIST=$(eval "$GREP_CMD")

# Check if any processes found
if [ -z "$PROCESS_LIST" ]; then
    echo "✅ No matching processes found."
    exit 0
fi

# Count processes
COUNT=$(echo "$PROCESS_LIST" | wc -l)

echo "Found $COUNT matching process(es):"
echo ""
echo "----------------------------------------"

# Display processes with line numbers
echo "$PROCESS_LIST" | nl -w3 -s'. '

echo "----------------------------------------"
echo ""

# Extract PIDs
PIDS=$(echo "$PROCESS_LIST" | awk '{print $2}')

echo "============================================"
# First verification: ask user to type "kill"
read -p "If you want to kill these processes, please enter [kill]: " confirmation

if [ "$confirmation" != "kill" ]; then
    echo ""
    echo "Operation cancelled. No processes were killed."
    exit 0
fi

# Second verification: ask y/N
echo ""
read -p "Do you want to kill these $COUNT process(es)? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Killing processes..."
    
    for pid in $PIDS; do
        if kill -9 $pid 2>/dev/null; then
            echo "  ✅ Killed process $pid"
        else
            echo "  ❌ Failed to kill process $pid (may already be terminated)"
        fi
    done
    
    echo ""
    echo "============================================"
    echo "✅ Process termination completed."
    echo "============================================"
else
    echo ""
    echo "Operation cancelled. No processes were killed."
fi

echo ""
