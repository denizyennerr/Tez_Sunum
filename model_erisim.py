import tensorflow as tf
from process import data_extraction
from costum_metric import BalancedAccuracy

###
# Instance
result = data_extraction.process_eeg_for_inference(
    file_path="chb01_03.edf",
    summary_file="uploaded/chb01/chb01-summary.txt",
    output_dir="inference_output",
    window_size=10.0,
    largest_window_ref=10.0,
)

X = result["X"]  # modele ver
y_true = result["y"]  # ground truth (varsa değerlendirme için)
##


##Load Model
model = tf.keras.models.load_model(
    filepath="model_paths/20260323-163632_0.5s/models/best_model_subject_chb01.keras",
    custom_objects={'BalancedAccuracy': BalancedAccuracy.BalancedAccuracy}  # ← This is the key line
)

model.summary()

###



import tensorflow as tf
import streamlit
import pyedflib
import sklearn
import fitz
import seaborn
import matplotlib
print('✅ TensorFlow:', tf.__version__)
print('✅ Streamlit:', streamlit.__version__)
print('✅ Protobuf:', tf.__version__ and '3.19.6')
print('✅ TÜM PAKETLER BAŞARIYLA İÇE AKTARILDI!')
