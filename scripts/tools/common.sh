#!/bin/sh
# ============================================================================
# Common Utilities Library
# ============================================================================
# Description: Shared functions and utilities for all scripts
# Usage: source scripts/tools/common.sh
# ============================================================================

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
	printf "${GREEN}[INFO]${NC} %s\n" "$1"
}

log_error() {
	printf "${RED}[ERROR]${NC} %s\n" "$1"
}

log_warn() {
	printf "${YELLOW}[WARN]${NC} %s\n" "$1"
}

log_step() {
	printf "${BLUE}[STEP]${NC} %s\n" "$1"
}

log_success() {
	printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
}

log_debug() {
	if [ "${DEBUG:-0}" = "1" ]; then
		printf "${CYAN}[DEBUG]${NC} %s\n" "$1"
	fi
}

# Error handling
die() {
	log_error "$1"
	exit "${2:-1}"
}

# Check if command exists
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

# Check if directory exists
dir_exists() {
	[ -d "$1" ]
}

# Check if file exists
file_exists() {
	[ -f "$1" ]
}

# Create directory if not exists
ensure_dir() {
	if ! dir_exists "$1"; then
		mkdir -p "$1" || die "Failed to create directory: $1"
	fi
}

# Separator line
print_separator() {
	printf "%s\n" "========================================="
}

# Section header
print_header() {
	echo ""
	print_separator
	printf "%s\n" "$1"
	print_separator
	echo ""
}

# Confirm action
confirm() {
	prompt="${1:-Are you sure?}"
	printf "${YELLOW}%s${NC} (y/N): " "$prompt"
	read -r response
	case "$response" in
	[yY][eE][sS] | [yY]) return 0 ;;
	*) return 1 ;;
	esac
}
