#!/bin/sh
# ============================================================================
# Version Bump Script
# ============================================================================
# Description: Bump version across multiple files and git tag
# Usage: bump.sh <new_version> [--dry-run] [--no-git]
# Example: scripts/bump.sh 0.2.0
#          scripts/bump.sh 0.2.0 --dry-run
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source common utilities
# shellcheck source=scripts/tools/common.sh
. "$SCRIPT_DIR/tools/common.sh"

# ============================================================================
# Configuration
# ============================================================================

VERSION_FILE="$PROJECT_ROOT/VERSION"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"
PROJECT_NAME="audex"
INIT_FILE="$PROJECT_ROOT/$PROJECT_NAME/__init__.py"
CONFIG_EXAMPLE_YML="$PROJECT_ROOT/.config.example.yml"
CONFIG_SYSTEM_LINUX="$PROJECT_ROOT/.config.system.linux.yml"
CONFIG_SYSTEM_WINDOWS="$PROJECT_ROOT/.config.system.windows.yml"
ENV_EXAMPLE_FILE="$PROJECT_ROOT/.env.example"

PYTHON="poetry run python"

# Flags
DRY_RUN=0
NO_GIT=0
NO_PUSH=0
NEW_VERSION=""

# ============================================================================
# Functions
# ============================================================================

# Parse arguments
parse_args() {
	while [ $# -gt 0 ]; do
		case "$1" in
		--dry-run)
			DRY_RUN=1
			log_info "Dry-run mode enabled"
			shift
			;;
		--no-git)
			NO_GIT=1
			log_info "Git operations disabled"
			shift
			;;
    --no-push)
      NO_PUSH=1
      log_info "Git push disabled"
      shift
      ;;
		-h | --help)
			show_usage
			exit 0
			;;
		*)
			if [ -z "$NEW_VERSION" ]; then
				NEW_VERSION="$1"
			else
				log_error "Unknown argument: $1"
				show_usage
				exit 1
			fi
			shift
			;;
		esac
	done

	if [ -z "$NEW_VERSION" ]; then
		log_error "Version number is required"
		show_usage
		exit 1
	fi
}

# Show usage
show_usage() {
	cat <<EOF
Usage: $(basename "$0") <new_version> [options]

Arguments:
  new_version           New version number (e.g., 0.2.0)

Options:
  --dry-run            Show what would be changed without making changes
  --no-git             Skip git operations (commit and tag)
  -h, --help           Show this help message

Examples:
  $(basename "$0") 0.2.0
  $(basename "$0") 0.2.0 --dry-run
  $(basename "$0") 0.2.0 --no-git

EOF
}

# Validate version format
validate_version() {
	version="$1"

	# Check semantic versioning format (X.Y.Z or X.Y.Z-suffix)
	if ! echo "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
		die "Invalid version format: $version (expected: X.Y.Z or X.Y.Z-suffix)"
	fi

	log_success "Version format validated: $version"
}

# Get current version from VERSION file
get_current_version() {
	if file_exists "$VERSION_FILE"; then
		current=$(cat "$VERSION_FILE" | tr -d '[:space:]')
		echo "$current"
	else
		echo "unknown"
	fi
}

# Update VERSION file
update_version_file() {
	new_version="$1"

	log_step "Updating VERSION file..."

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would write '$new_version' to $VERSION_FILE"
		return 0
	fi

	echo "$new_version" >"$VERSION_FILE"
	log_success "✓ Updated: $VERSION_FILE"
}

# Update pyproject.toml
update_pyproject() {
	new_version="$1"

	log_step "Updating pyproject.toml..."

	if ! file_exists "$PYPROJECT_FILE"; then
		log_warn "File not found: $PYPROJECT_FILE"
		return 1
	fi

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would update version in $PYPROJECT_FILE"
		return 0
	fi

	# Update version in [tool.poetry] section
	if grep -q '^\[tool\.poetry\]' "$PYPROJECT_FILE"; then
		# Use sed to update version under [tool.poetry]
		sed -i.bak '/^\[tool\.poetry\]/,/^\[/ s/^version = .*/version = "'"$new_version"'"/' "$PYPROJECT_FILE"
		rm -f "$PYPROJECT_FILE.bak"
		log_success "✓ Updated: $PYPROJECT_FILE ([tool.poetry] section)"
	else
		log_warn "Section [tool.poetry] not found in $PYPROJECT_FILE"
	fi

	# Also update [project] section if it exists
	if grep -q '^\[project\]' "$PYPROJECT_FILE"; then
		sed -i.bak '/^\[project\]/,/^\[/ s/^version = .*/version = "'"$new_version"'"/' "$PYPROJECT_FILE"
		rm -f "$PYPROJECT_FILE.bak"
		log_success "✓ Updated: $PYPROJECT_FILE ([project] section)"
	fi
}

# Update __init__.py
update_init_py() {
	new_version="$1"

	log_step "Updating $PROJECT_NAME/__init__.py..."

	if ! file_exists "$INIT_FILE"; then
		log_warn "File not found: $INIT_FILE"
		return 1
	fi

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would update __version__ in $INIT_FILE"
		return 0
	fi

	# Update __version__ variable
	sed -i.bak 's/^__version__ = .*/__version__ = "'"$new_version"'"/' "$INIT_FILE"
	rm -f "$INIT_FILE.bak"
	log_success "✓ Updated: $INIT_FILE"
}

# Update configuration files
update_config_files() {
  log_step "Updating configuration files..."

  if [ $DRY_RUN -eq 1 ]; then
    log_info "[DRY-RUN] Would update $CONFIG_EXAMPLE_YML, $CONFIG_SYSTEM_LINUX, $CONFIG_SYSTEM_WINDOWS, $ENV_EXAMPLE_FILE"
    return 0
  fi

  echo y | $PYTHON -m audex init gencfg -o $CONFIG_EXAMPLE_YML --format yaml
  log_success "✓ Updated: $CONFIG_EXAMPLE_YML"

  echo y | $PYTHON -m audex init gencfg -o $CONFIG_SYSTEM_LINUX --format system --platform linux
  log_success "✓ Updated: $CONFIG_SYSTEM_LINUX"

  echo y | $PYTHON -m audex init gencfg -o $CONFIG_SYSTEM_WINDOWS --format system --platform windows
  log_success "✓ Updated: $CONFIG_SYSTEM_WINDOWS"

  echo y | $PYTHON -m audex init gencfg -o $ENV_EXAMPLE_FILE --format dotenv
  log_success "✓ Updated: $ENV_EXAMPLE_FILE"
}

# Create git commit
create_git_commit() {
	new_version="$1"

	if [ $NO_GIT -eq 1 ]; then
		log_info "Skipping git commit (--no-git flag)"
		return 0
	fi

	if ! command_exists git; then
		log_warn "Git is not installed, skipping git operations"
		return 1
	fi

	# Check if we're in a git repository
	if ! git rev-parse --git-dir >/dev/null 2>&1; then
		log_warn "Not a git repository, skipping git operations"
		return 1
	fi

	log_step "Creating git commit..."

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would commit with message: 'chore: bump version to $new_version'"
		return 0
	fi

	# Add files
	git add "$VERSION_FILE" "$PYPROJECT_FILE" "$INIT_FILE" 2>/dev/null || true

	# Create commit
	if git diff --staged --quiet; then
		log_warn "No changes to commit"
	else
		git commit -m "chore: bump version to $new_version"
		log_success "✓ Git commit created"
	fi
}

# Create git tag
create_git_tag() {
	new_version="$1"

	if [ $NO_GIT -eq 1 ]; then
		log_info "Skipping git tag (--no-git flag)"
		return 0
	fi

	if ! command_exists git; then
		log_warn "Git is not installed, skipping git tag"
		return 1
	fi

	if ! git rev-parse --git-dir >/dev/null 2>&1; then
		log_warn "Not a git repository, skipping git tag"
		return 1
	fi

	log_step "Creating git tag..."

	tag_name="v$new_version"

	# Check if tag already exists
	if git rev-parse "$tag_name" >/dev/null 2>&1; then
		log_warn "Tag $tag_name already exists"
		if [ $DRY_RUN -eq 0 ]; then
			if confirm "Do you want to delete and recreate the tag?"; then
				git tag -d "$tag_name"
				log_info "Deleted existing tag: $tag_name"
			else
				return 0
			fi
		else
			return 0
		fi
	fi

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would create tag: $tag_name"
		return 0
	fi

	# Create annotated tag
	git tag -a "$tag_name" -m "Release version $new_version"
	log_success "✓ Git tag created: $tag_name"
	echo ""
	log_info "To push the tag, run: git push origin $tag_name"
}

# Push git tag
push_git_tag() {
	new_version="$1"

  if [ $NO_GIT -eq 1 ] || [ $NO_PUSH -eq 1 ]; then
    log_info "Skipping git push"
    return 0
  fi

	if ! command_exists git; then
		return 1
	fi

	if ! git rev-parse --git-dir >/dev/null 2>&1; then
		return 1
	fi

	log_step "Pushing to remote..."

	tag_name="v$new_version"

	if [ $DRY_RUN -eq 1 ]; then
		log_info "[DRY-RUN] Would push: git push origin main"
		log_info "[DRY-RUN] Would push: git push origin $tag_name"
		return 0
	fi

	# Ask for confirmation
	if ! confirm "Do you want to push commit and tag to remote?"; then
		log_info "Skipping push. Run manually:"
		echo "  git push origin main"
		echo "  git push origin $tag_name"
		return 0
	fi

	# Push commit
	if git push origin "$(git branch --show-current)" 2>/dev/null; then
		log_success "✓ Pushed commit to remote"
	else
		log_error "Failed to push commit"
		return 1
	fi

	# Push tag
	if git push origin "$tag_name" 2>/dev/null; then
		log_success "✓ Pushed tag to remote: $tag_name"
	else
		log_error "Failed to push tag"
		return 1
	fi
}

# Verify all files were updated
verify_updates() {
	new_version="$1"

	if [ $DRY_RUN -eq 1 ]; then
		return 0
	fi

	log_step "Verifying updates..."

	errors=0

	# Check VERSION file
	if file_exists "$VERSION_FILE"; then
		current=$(cat "$VERSION_FILE" | tr -d '[:space:]')
		if [ "$current" = "$new_version" ]; then
			log_success "✓ VERSION file: $current"
		else
			log_error "✗ VERSION file: expected $new_version, found $current"
			errors=$((errors + 1))
		fi
	fi

	# Check pyproject.toml
	if file_exists "$PYPROJECT_FILE"; then
		if grep -q "version = \"$new_version\"" "$PYPROJECT_FILE"; then
			log_success "✓ pyproject.toml: $new_version"
		else
			log_error "✗ pyproject.toml: version not found or incorrect"
			errors=$((errors + 1))
		fi
	fi

	# Check __init__.py
	if file_exists "$INIT_FILE"; then
		if grep -q "__version__ = \"$new_version\"" "$INIT_FILE"; then
			log_success "✓ __init__.py: $new_version"
		else
			log_error "✗ __init__.py: version not found or incorrect"
			errors=$((errors + 1))
		fi
	fi

	if [ $errors -gt 0 ]; then
		echo ""
		log_error "Verification failed with $errors error(s)"
		return 1
	fi

	echo ""
	log_success "All files verified successfully"
}

# Show summary
show_summary() {
	old_version="$1"
	new_version="$2"

	print_separator
	log_info "Version Bump Summary:"
	echo ""
	echo "  Old Version: $old_version"
	echo "  New Version: $new_version"
	echo ""
	echo "  Updated files:"
	echo "    - $VERSION_FILE"
	echo "    - $PYPROJECT_FILE"
	echo "    - $INIT_FILE"
	echo ""
	if [ $NO_GIT -eq 0 ]; then
		echo "  Git operations:"
		echo "    - Commit created"
		echo "    - Tag created: v$new_version"
		echo ""
		log_info "Next steps:"
		echo "    git push origin main"
		echo "    git push origin v$new_version"
	fi
	print_separator
}

# ============================================================================
# Main
# ============================================================================

main() {
	print_header "Version Bump"

	# Parse arguments
	parse_args "$@"

	# Validate version format
	validate_version "$NEW_VERSION"
	echo ""

	# Get current version
	current_version=$(get_current_version)
	log_info "Current version: $current_version"
	log_info "New version: $NEW_VERSION"
	echo ""

	# Confirm if not dry-run
	if [ $DRY_RUN -eq 0 ]; then
		if ! confirm "Do you want to proceed with version bump?"; then
			log_info "Version bump cancelled"
			exit 0
		fi
		echo ""
	fi

	# Update files
	update_version_file "$NEW_VERSION"
	update_pyproject "$NEW_VERSION"
	update_init_py "$NEW_VERSION"
	update_config_files
	echo ""

	# Verify updates
	verify_updates "$NEW_VERSION"

	# Git operations
	if [ $DRY_RUN -eq 0 ]; then
		echo ""
		create_git_commit "$NEW_VERSION"
		create_git_tag "$NEW_VERSION"
		push_git_tag "$NEW_VERSION"
	fi

	# Show summary
	echo ""
	if [ $DRY_RUN -eq 1 ]; then
		log_info "Dry-run complete. No files were modified."
	else
		show_summary "$current_version" "$NEW_VERSION"
		log_success "Version bump complete!"
	fi
}

main "$@"
