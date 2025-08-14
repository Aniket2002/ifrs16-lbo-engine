"""
Commit arXiv Preparation Files

This script commits all the arXiv preparation work to git
and prepares for v1.0.0 release tagging.
"""

import subprocess
import sys

def run_git_command(cmd):
    """Run git command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"   Error: {e.stderr.strip()}")
        return False

def main():
    print("📝 COMMITTING ARXIV PREPARATION FILES")
    print("=" * 50)
    
    # Stage all new arXiv files
    arxiv_files = [
        "paper/",
        "mathematical_appendix.tex",
        "theoretical_assumptions.tex",
        "theoretical_config.py",
        "theoretical_guarantees.py",
        "benchmark_creation.py",
        "breakthrough_pipeline.py",
        "failure_mode_analysis.py",
        "generate_arxiv_figures.py",
        "prepare_arxiv_submission.py",
        "validate_arxiv_submission.py",
        "CITATION.cff",
        "requirements-dev.txt", 
        "Dockerfile",
        "pytest.ini",
        ".github/workflows/ci.yml",
        "tests/test_validation.py",
        "Makefile",
        "PUBLICATION_READINESS.md",
        "FINAL_ARXIV_CHECKLIST.md",
        "ARXIV_SUBMISSION_READY.md"
    ]
    
    print("📋 Staging arXiv preparation files...")
    for file in arxiv_files:
        run_git_command(f"git add {file}")
    
    # Commit with descriptive message
    commit_msg = "feat: Complete arXiv submission preparation\n\n- Add complete LaTeX manuscript with formal proofs\n- Implement theoretical guarantees with empirical validation\n- Create public benchmark dataset for community use\n- Add reproducible build pipeline with Docker\n- Include comprehensive test suite with CI/CD\n- Generate figures with git hash stamps for reproducibility\n- Prepare submission archive with all required files\n\nReady for arXiv submission to q-fin.GN (primary), stat.ME (secondary)"
    
    print("\n📝 Committing arXiv preparation...")
    success = run_git_command(f'git commit -m "{commit_msg}"')
    
    if success:
        print("\n🎯 Ready for release tagging!")
        print("Next steps:")
        print("1. git tag -a v1.0.0 -m 'Release v1.0.0: arXiv submission ready'")
        print("2. python prepare_arxiv_submission.py")
        print("3. Submit to arXiv with generated archive")
        return 0
    else:
        print("\n❌ Commit failed - please check git status")
        return 1

if __name__ == "__main__":
    exit(main())
