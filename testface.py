from deepface import DeepFace

def check_same_person(img1_path, img2_path):
    try:
        # verify() performs detection, alignment, and comparison
        # model_name="Facenet512" is highly accurate and stable on M2
        result = DeepFace.verify(
            img1_path = img1_path, 
            img2_path = img2_path,
            model_name = "Facenet512",
            detector_backend = "opencv"
        )
        print(result)
        print(f"Is it the same person? {result['verified']}")
        print(f"Distance (Similarity): {result['distance']}")
        print(f"Threshold used: {result['threshold']}")
        
    except Exception as e:
        print(f"Error: {e}")

# Usage
check_same_person("face1/ID_2.jpg", "face1/Selfie_13.jpg")
