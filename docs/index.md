<link href="style.css" rel="stylesheet"/>

<h1>About the generation of synthetic laparascopic images using diffusion-based models</h1>

<div class="row">
  <figure>
    <h4>Dall-e2</h4>
    <img src="./assets/Dalle2/dalle2_3_T45-grasper%20retract%20gallbladder%20in%20preparation.png" alt="Dall-e2_3_CholecT45" width='250'>
    <figcaption>"grasper retract gallbladder in preparation"</figcaption>
  </figure>
  <figure>
    <h4>Imagen</h4>
    <img src="./assets/Imagen/Imagen_7_T45-grasper%20grasp%20gallbladder%20and%20grasper%20retract%20gallbladder%20and%20hook%20dissect%20gallbladder%20in%20calot%20triangle%20dissection.png" alt="Imagen_7_CholecT45" width='250'>
    <figcaption>"grasper grasp gallbladder and grasper retract gallbladder and hook dissect gallbladder in calot triangle dissection"</figcaption>
  </figure>
  <figure>
    <h4>Elucidated Imagen</h4>
    <img src="./assets/EluciatedImagen/ElucidatedImagen_5_T45-grasper%20retract%20gallbladder%20and%20grasper%20retract%20omentum%20and%20hook%20dissect%20omentum%20in%20calot%20triangle%20dissection.png" alt="Dall-e2" width='250'>
    <figcaption>"grasper retract gallbladder and hook dissect gallbladder in calot triangle dissection"</figcaption>
  </figure>
</div>



<h3> [Navigating the Synthetic Realm: Harnessing Diffusion-based Models for Laparoscopic Text-to-Image Generation](https://arxiv.org/abs/2312.03043) <h3>

<h4>Simeon Allmendinger, Patrick Hemmer, Niklas Kühl, Moritz Queisner, Igor Sauer, Leopold Müller, Johannes Jakubik, Michael Vössing </h4>

<p> Recent advances in synthetic imaging open up opportunities for obtaining additional data in the field of surgical imaging. This data can provide reliable supplements supporting surgical applications and decision-making through computer vision. Particularly the field of image-guided surgery, such as laparoscopic and robotic-assisted surgery, benefits strongly from synthetic image datasets and virtual surgical training methods. Our study presents an intuitive approach for generating synthetic laparoscopic images from short text prompts using diffusion-based generative models. We demonstrate the usage of state-of-the-art text-to-image architectures in the context of laparoscopic imaging with regard to the surgical removal of the gallbladder as an example. Results on fidelity and diversity demonstrate that diffusion-based models can acquire knowledge about the style and semantics in the field of image-guided surgery. A validation study with a human assessment survey underlines the realistic nature of our synthetic data, as medical personnel detects actual images in a pool with generated images causing a false-positive rate of 66%. In addition, the investigation of a state-of-the-art machine learning model to recognize surgical actions indicates enhanced results when trained with additional generated images of up to 5.20%. Overall, the achieved image quality contributes to the usage of computer-generated images in surgical applications and enhances its path to maturity.</p>