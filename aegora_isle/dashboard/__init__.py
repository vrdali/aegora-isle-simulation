"""Streamlit observability dashboard (spec §7).

Launched per spec Table 9::

    streamlit run aegora_isle/dashboard/app.py

The UI layer lives in ``app.py``; everything testable lives in
``data_access.py`` and ``process_control.py``, neither of which imports
Streamlit.
"""
