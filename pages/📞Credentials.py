import streamlit as st
import base64

# --- VERİLER ---
NAME = "Deniz Yener"
TITLE = ""
PROFILE_PIC = "foto.jpg"

EDUCATION = {
    "University": "Tallinn University, Tampere University, Lusófona University",
    "Masters": "Artificial Intelligence for Sustainable Societies - Erasmus Mundus Joint Master Degree"
}

CONTACT = {
    "Email": "deniz.yener9@outlook.com",
    "Phone": "+358458628078"
}

SOCIAL_LINKS = {
    "LinkedIn": "https://www.linkedin.com/in/denizyennerr/",
    "GitHub": "https://github.com/denizyennerr",
    "Portfolio": "https://www.denizportfoliowebsite.com/"
}


# --- YARDIMCI FONKSİYON ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


# --- SAYFA DÜZENİ ---
st.title("👨‍💻 Profil ve İletişim Bilgileri")
st.markdown("---")

# st.container(border=True) Streamlit 1.22'de yok → CSS ile kart efekti veriyoruz
st.markdown("""
<style>
.profile-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2.5])

with col1:
    img_base64 = get_base64_of_bin_file(PROFILE_PIC)
    if img_base64:
        st.markdown(
            f'''
            <div style="display: flex; justify-content: center; align-items: flex-start; padding-top: 0px;">
                <img src="data:image/jpeg;base64,{img_base64}" 
                     style="border-radius: 50%; width: 100%; max-width: 200px; aspect-ratio: 1/1; object-fit: cover;
                            box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"⚠️ Fotoğraf bulunamadı! '{PROFILE_PIC}' dosyasını ekleyin.")

with col2:
    st.markdown(f'<div class="profile-card"><p style="margin:0; font-size:38px; font-weight:600; color:white;">{NAME}</p></div>', unsafe_allow_html=True)
    # Name frame is now above title
    st.markdown(f"<h4 style='margin-top: 5px; color: #666; font-weight: 400;'>{TITLE}</h4>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    info_col1, info_col2 = st.columns([1, 1])

    with info_col1:
        st.markdown("### 🎓")
        st.markdown(f"**University:**<br>{EDUCATION['University']}", unsafe_allow_html=True)
        st.markdown(f"**Masters:**<br>{EDUCATION['Masters']}", unsafe_allow_html=True)

    with info_col2:
        st.markdown("### 📞")
        st.markdown(f"📧 **E-mail:** [{CONTACT['Email']}](mailto:{CONTACT['Email']})")
        st.markdown(f"📱 **Phone:** {CONTACT['Phone']}")

        st.markdown("### 🔗")
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; margin-top: 5px;">
                <a href="{SOCIAL_LINKS['LinkedIn']}" target="_blank"
                   style="text-decoration: none; font-size: 16px; color: white; font-weight: 600;">
                    🔵 LinkedIn
                </a>
                <a href="{SOCIAL_LINKS['GitHub']}" target="_blank"
                   style="text-decoration: none; font-size: 16px; color: white; font-weight: 600;">
                    ⚫ GitHub
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")