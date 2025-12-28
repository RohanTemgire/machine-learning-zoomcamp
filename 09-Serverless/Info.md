# To deploy a regression model
1. Create a docker file/image [use uv package manager or anything of your choice]
2. Load this docker file to Amazon ECR
3. Create a lambda function. [Select "container image" while creating the function. then choose the ECR container that you created in the previous step]
4. Once the Image is built. When you test the lambda function for the first time, it will take time to load. Since loading 2GB models takes time. Once the model is loaded fully, post that the responses would be quicker. within milliseconds

# To deploy a Tensorflow model
#### We use Onnx here because it converts various runtimes to single onnx runtime and than deployed on cloud
<img src="https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/b1-thumbnail_webp-600x300.webp">
1. Change the keras model to saved_model() type, in onnx type
2. 