// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';
//import {NDArrayMathGPU, Array3D, ENV}from 'deeplearn';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

// Initiate variables
var infoTexts = [];
var training = -1; // -1 when no class is being trained
var videoPlaying = false;

// Initiate deeplearn.js math and knn classifier objects
var knn = new knn_image_classifier.KNNImageClassifier(NUM_CLASSES, TOPK, deeplearn.ENV.math);

// Create video element that will contain the webcam image
var video = document.createElement('video');
video.setAttribute('autoplay', '');
video.setAttribute('playsinline', '');

// Add video element to DOM
document.body.appendChild(video);

var timer;

var labelToClass = {};
var labels = [];

var confidences = {};

var topChoice;

// Create training buttons and info texts
for(let i=0;i<NUM_CLASSES; i++){
  const div = document.createElement('div');
  document.body.appendChild(div);
  div.style.marginBottom = '10px';

  // Create training button
  const button = document.createElement('button')
  button.innerText = "Train "+i;
  div.appendChild(button);

  // Listen for mouse events when clicking the button
  button.addEventListener('mousedown', () => training = i);
  button.addEventListener('mouseup', () => training = -1);

  // Create info text
  const infoText = document.createElement('span')
  infoText.innerText = " No examples added";
  div.appendChild(infoText);
  infoTexts.push(infoText);
}


// Setup webcam
navigator.mediaDevices.getUserMedia({video: true, audio: false})
.then((stream) => {
  video.srcObject = stream;
  video.width = IMAGE_SIZE;
  video.height = IMAGE_SIZE;

  video.addEventListener('playing', ()=> videoPlaying = true);
  video.addEventListener('paused', ()=> videoPlaying = false);
}).catch((err) => {
  console.log("error");
  console.log(err);
});

// Load knn model
knn.load()
.then(() => start());

function start(){
  if (timer) {
    stop();
  }
  video.play();
  timer = requestAnimationFrame(animate.bind(this));
}

function stop(){
  video.pause();
  cancelAnimationFrame(timer);
}

function animate(){
  if(videoPlaying){
    // Get image data from video element
    const image = deeplearn.Array3D.fromPixels(video);

    // Train class if one of the buttons is held down
    if(training != -1){
      // Add current image to classifier
      knn.addImage(image, training)
    }

    // If any examples have been added, run predict
    const exampleCount = knn.getClassExampleCount();
    if(Math.max(...exampleCount) > 0){
      knn.predictClass(image)
      .then((res)=>{
        for(let i=0;i<NUM_CLASSES; i++){
          // Make the predicted class bold
          if(res.classIndex == i){
            topChoice = i;
            infoTexts[i].style.fontWeight = 'bold';
          } else {
            infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if(exampleCount[i] > 0){
            confidences[i] = res.confidences[i];
            infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i]*100}%`
          }
        }
      })
      // Dispose image when done
      .then(()=> image.dispose())
    } else {
      image.dispose()
    }
  }
  timer = requestAnimationFrame(animate.bind(this));
}

function startTraining(label) {
  var numClasses = Object.keys(labelToClass).length;
  if (!labelToClass.hasOwnProperty(label)) {
    if (numClasses == NUM_CLASSES) {
      return;
    }
    labelToClass[label] = numClasses;
    labels.push(label);
  }
  training = labelToClass[label];
}

function stopTraining() {
  training = -1;
}

function getSampleCount(label) {
  if (!labelToClass.hasOwnProperty(label)) {
    return -1;
  }
  var counts = knn.getClassExampleCount();
  return counts[labelToClass[label]];
}

function getConfidence(label) {
  if (!labelToClass.hasOwnProperty(label)) {
    return -1;
  }
  return confidences[labelToClass[label]];
}

function getClassification() {
  return labels[topChoice];
}
