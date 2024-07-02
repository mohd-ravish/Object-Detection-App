import React, { useRef, useState } from "react";
import "@tensorflow/tfjs-backend-cpu";
//import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

function ObjectDetector() {
  const fileInputRef = useRef();
  const imageRef = useRef();
  const containerRef = useRef();
  const [imgData, setImgData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setLoading] = useState(false);

  const isEmptyPredictions = !predictions || predictions.length === 0;

  const openFilePicker = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const normalizePredictions = (predictions, imgSize, containerSize) => {
    if (!predictions || !imgSize || !imageRef || !containerSize) return predictions || [];
    return predictions.map((prediction) => {
      const { bbox } = prediction;
      const oldX = bbox[0];
      const oldY = bbox[1];
      const oldWidth = bbox[2];
      const oldHeight = bbox[3];

      const x = (oldX * containerSize.width) / imgSize.width;
      const y = (oldY * containerSize.height) / imgSize.height;
      const width = (oldWidth * containerSize.width) / imgSize.width;
      const height = (oldHeight * containerSize.height) / imgSize.height;

      return { ...prediction, bbox: [x, y, width, height] };
    });
  };

  const detectObjectsOnImage = async (imageElement, imgSize) => {
    const model = await cocoSsd.load({});
    const predictions = await model.detect(imageElement, 10);
    const containerSize = containerRef.current.getBoundingClientRect();
    const normalizedPredictions = normalizePredictions(predictions, imgSize, containerSize);
    setPredictions(normalizedPredictions);
    console.log("Predictions: ", predictions);
  };

  const readImage = (file) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.onload = () => resolve(fileReader.result);
      fileReader.onerror = () => reject(fileReader.error);
      fileReader.readAsDataURL(file);
    });
  };

  const onSelectImage = async (e) => {
    setPredictions([]);
    setLoading(true);

    const file = e.target.files[0];
    const imgData = await readImage(file);
    setImgData(imgData);

    const imageElement = document.createElement("img");
    imageElement.src = imgData;

    imageElement.onload = async () => {
      const imgSize = {
        width: imageElement.width,
        height: imageElement.height,
      };
      await detectObjectsOnImage(imageElement, imgSize);
      setLoading(false);
    };
  };

  return (
    <div className="object-detector-container">
      <h1>Object Detection App</h1>
      <div className="detector-container" ref={containerRef}>
        {imgData && <img src={imgData} ref={imageRef} className="target-img" />}
        {!isEmptyPredictions &&
          predictions.map((prediction, idx) => (
            <div
              key={idx}
              className="target-box"
              style={{
                left: `${prediction.bbox[0]}px`,
                top: `${prediction.bbox[1]}px`,
                width: `${prediction.bbox[2]}px`,
                height: `${prediction.bbox[3]}px`,
              }}
              data-class={prediction.class}
              data-score={(prediction.score * 100).toFixed(1)}
            />
          ))}
      </div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={onSelectImage}
        className="hidden-file-input"
      />
      <button onClick={openFilePicker} className="select-button">
        {isLoading ? "Recognizing..." : "Select Image"}
      </button>
    </div>
  );
}

export default ObjectDetector;