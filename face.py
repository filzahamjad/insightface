from realutils.face.insightface import isf_face_batch_similarity, isf_analysis_faces, isf_faces_visualize

image_path = "face1"
# get the analysis all the faces
faces = isf_analysis_faces(image_path)
print(faces)

# compare them
print(isf_face_batch_similarity([face.embedding for face in faces]))

# visualize it
isf_faces_visualize(image_path, faces).show()
