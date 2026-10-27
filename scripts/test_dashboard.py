#!/usr/bin/env python3
"""
Test script for the FedXplain-IoMT Streamlit dashboard.
Run with: python scripts/test_dashboard.py
"""
import os
import sys
import subprocess
from streamlit.testing.v1 import AppTest

def main():
    # Change to repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"[INFO] Changed to repo root: {repo_root}")

    # Path to app.py
    app_path = os.path.join(repo_root, "app.py")
    if not os.path.exists(app_path):
        print(f"[ERROR] app.py not found at {app_path}")
        sys.exit(1)

    # Load the app with AppTest
    print("[INFO] Loading app with AppTest...")
    at = AppTest.from_file(app_path, default_timeout=300)

    # Initial run
    print("[INFO] Running initial app execution...")
    at.run()

    # Check for exceptions
    if at.exception:
        print(f"[FAIL] App threw an exception during initial run:")
        print(at.exception)
        sys.exit(1)
    else:
        print("[OK] No exceptions during initial run.")

    # Check that we have 6 tabs
    tabs_attr = getattr(at, "tabs", None)
    if tabs_attr is None:
        print("[WARN] 'tabs' attribute not found on AppTest object; using fallback method.")
        # Fallback: try to count tab-like elements in at.main
        tab_count = 0
        for el in at.main:
            # Look for elements that might be tabs (this is heuristic)
            if hasattr(el, 'name') and el.name == 'stTabs':
                # Try to get child tabs
                if hasattr(el, 'children'):
                    tab_count = len(el.children)
                break
        if tab_count != 6:
            print(f"[FAIL] Expected 6 tabs, found {tab_count} via fallback.")
            sys.exit(1)
        else:
            print(f"[OK] Found 6 tabs via fallback.")
    else:
        if len(tabs_attr) != 6:
            print(f"[FAIL] Expected 6 tabs, found {len(tabs_attr)}.")
            sys.exit(1)
        else:
            print("[OK] Found 6 tabs.")

    # Now test the cluster_fl placeholder by setting the div_method selectbox to "cluster_fl"
    print("[INFO] Testing cluster_fl placeholder selection...")
    try:
        # Find the selectbox with key="div_method"
        div_method_sb = None
        for sb in at.selectbox:
            if sb.key == "div_method":
                div_method_sb = sb
                break
        if div_method_sb is None:
            print("[WARN] Could not find selectbox with key='div_method' via attribute; searching manually.")
            # Manual search
            for el in at.main:
                if hasattr(el, 'key') and el.key == "div_method" and hasattr(el, 'type') and el.type == "selectbox":
                    div_method_sb = el
                    break
        if div_method_sb is None:
            print("[FAIL] Could not locate div_method selectbox.")
            sys.exit(1)

        # Set value to "cluster_fl"
        div_method_sb.set_value("cluster_fl")
        # Re-run
        at.run()

        # Check for exceptions after setting cluster_fl
        if at.exception:
            print(f"[FAIL] App threw an exception after setting div_method to cluster_fl:")
            print(at.exception)
            sys.exit(1)
        else:
            print("[OK] No exceptions after setting cluster_fl.")

        # Check that cluster_fl renders as REAL results (not placeholder charts).
        placeholder_found = False
        placeholder_texts = [
            "slot empty - cluster_fl results pending",
            "in progress - port pending",
            "Cluster-FL (pending)",
            "Top‑10 SHAP values (placeholder)",
            "Jaccard similarity heatmap (placeholder)",
        ]
        # Search in all elements' values and labels
        for el in at.main:
            # Check value attribute
            if hasattr(el, 'value'):
                val = str(el.value)
                if any(text in val for text in placeholder_texts):
                    placeholder_found = True
                    break
            # Check label attribute
            if hasattr(el, 'label'):
                lbl = str(el.label)
                if any(text in lbl for text in placeholder_texts):
                    placeholder_found = True
                    break
            # Check text attribute (for markdown, etc.)
            if hasattr(el, 'text'):
                txt = str(el.text)
                if any(text in txt for text in placeholder_texts):
                    placeholder_found = True
                    break

        if placeholder_found:
            print("[FAIL] Placeholder text found for cluster_fl, but results are merged "
                  "(divergence_summary.cluster_fl is populated in study_results.json).")
            sys.exit(1)
        else:
            print("[OK] No placeholder text for cluster_fl; real results render.")

    except Exception as e:
        print(f"[FAIL] Error during cluster_fl test: {e}")
        sys.exit(1)

    # Count dataframes (optional, just for info)
    df_count = len(at.dataframe)
    print(f"[INFO] Found {df_count} dataframes in the app.")

    print("\n[SUCCESS] All tests passed:")
    print("  - 0 exceptions during initial run")
    print("  - 6 tabs present")
    print("  - 0 exceptions after setting cluster_fl")
    print("  - cluster_fl renders real results (no placeholder)")

    # Exit with success
    sys.exit(0)

if __name__ == "__main__":
    main()