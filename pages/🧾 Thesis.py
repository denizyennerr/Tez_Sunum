import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF

PDF_PATH = "presentation/Master_Thesis.pdf"

# --- SESSION STATE ---
# 'thesis_page' key kullanıyoruz — 💻 Presentation.py ile çakışmaması için
if "thesis_page" not in st.session_state:
    st.session_state.thesis_page = 1

# --- QUERY PARAM SYNC (Streamlit 1.22 API) ---
params = st.experimental_get_query_params()
if "thesis_page" in params:
    try:
        st.session_state.thesis_page = int(params["thesis_page"][0])
    except (ValueError, IndexError):
        pass


# --- LOAD PDF ---
@st.cache_resource
def load_pdf(path):
    try:
        return fitz.open(path)
    except Exception:
        return None


doc = load_pdf(PDF_PATH)

if doc is None:
    st.error(f"⚠️ PDF file not found! Please check the file path: {PDF_PATH}")
    st.stop()

total_pages = len(doc)

# --- TOP BAR ---
col_title, col_toggle = st.columns([4, 1])

with col_title:
    st.title("🧾 Thesis Viewer")

with col_toggle:
    st.write("")
    st.write("")
    # st.toggle Streamlit 1.22'de yok → st.checkbox kullanıyoruz
    fullscreen = st.checkbox("🖥️ Fullscreen", value=False)

with st.expander("⚙️ Quick Navigation & Settings"):
    page_number = st.slider("Quick Page Selection", 1, total_pages, st.session_state.thesis_page)
    if page_number != st.session_state.thesis_page:
        st.session_state.thesis_page = page_number
        st.experimental_set_query_params(thesis_page=page_number)
        st.experimental_rerun()

st.divider()

# --- RENDER PAGE ---
page = doc[st.session_state.thesis_page - 1]
render_quality = 4.0
mat = fitz.Matrix(render_quality, render_quality)
pix = page.get_pixmap(matrix=mat)

if fullscreen:
    spacer_left, img_col, spacer_right = st.columns([0.5, 4, 0.5])
else:
    spacer_left, img_col, spacer_right = st.columns([1.5, 2, 1.5])

with img_col:
    st.image(pix.tobytes("png"), use_column_width=True)

st.divider()

# --- NAVIGATION BUTTONS ---
ctrl1, ctrl2, ctrl3 = st.columns([1, 2, 1])

with ctrl1:
    if st.button("⬅️ Previous", use_container_width=True):
        if st.session_state.thesis_page > 1:
            st.session_state.thesis_page -= 1
            st.experimental_set_query_params(thesis_page=st.session_state.thesis_page)
            st.experimental_rerun()

with ctrl2:
    st.markdown(
        f"<div style='text-align:center; font-size:22px; font-weight:600; color:#4F8BF9; padding-top:5px;'>"
        f"Page {st.session_state.thesis_page} / {total_pages}"
        f"</div>",
        unsafe_allow_html=True
    )

with ctrl3:
    if st.button("Next ➡️", use_container_width=True):
        if st.session_state.thesis_page < total_pages:
            st.session_state.thesis_page += 1
            st.experimental_set_query_params(thesis_page=st.session_state.thesis_page)
            st.experimental_rerun()

# --- KEYBOARD NAVIGATION ---
components.html("""
<script>
    const parentDoc = window.parent.document;

    if (parentDoc.pdfKeydownHandlerThesis) {
        parentDoc.removeEventListener('keydown', parentDoc.pdfKeydownHandlerThesis);
    }

    parentDoc.pdfKeydownHandlerThesis = function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        const buttons = Array.from(parentDoc.querySelectorAll('button'));
        const prevBtn = buttons.find(b => b.innerText.includes('Previous'));
        const nextBtn = buttons.find(b => b.innerText.includes('Next'));

        if (e.key === "ArrowLeft" && prevBtn) prevBtn.click();
        if (e.key === "ArrowRight" && nextBtn) nextBtn.click();
    };

    parentDoc.addEventListener('keydown', parentDoc.pdfKeydownHandlerThesis);
</script>
""", height=0)