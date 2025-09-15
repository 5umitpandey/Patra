import * as tf from '@tensorflow/tfjs';

export class ImageAnalysisEngine {
  private model: tf.LayersModel | null = null;
  private isInitialized = false;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel(): Promise<void> {
    try {
      // Create a simple CNN model for image quality assessment
      this.model = tf.sequential({
        layers: [
          tf.layers.conv2d({
            inputShape: [224, 224, 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu'
          }),
          tf.layers.maxPooling2d({ poolSize: 2 }),
          tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
          tf.layers.maxPooling2d({ poolSize: 2 }),
          tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
          tf.layers.flatten(),
          tf.layers.dense({ units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.5 }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' }) // Quality score 0-1
        ]
      });

      this.model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      this.isInitialized = true;
    } catch (error) {
      console.warn('Failed to initialize image analysis model:', error);
      this.isInitialized = false;
    }
  }

  public async analyzePhotoQuality(imageUrl: string): Promise<{
    qualityScore: number;
    brightness: number;
    contrast: number;
    sharpness: number;
    composition: number;
  }> {
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      return new Promise((resolve) => {
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d')!;
          
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const analysis = this.analyzeImageData(imageData);
          
          resolve(analysis);
        };
        
        img.onerror = () => {
          // Return default values if image fails to load
          resolve({
            qualityScore: 0.7,
            brightness: 0.5,
            contrast: 0.5,
            sharpness: 0.5,
            composition: 0.5
          });
        };
        
        img.src = imageUrl;
      });
    } catch (error) {
      console.warn('Image analysis failed:', error);
      return {
        qualityScore: 0.7,
        brightness: 0.5,
        contrast: 0.5,
        sharpness: 0.5,
        composition: 0.5
      };
    }
  }

  private analyzeImageData(imageData: ImageData): {
    qualityScore: number;
    brightness: number;
    contrast: number;
    sharpness: number;
    composition: number;
  } {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    // Calculate brightness
    let totalBrightness = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      totalBrightness += (r + g + b) / 3;
    }
    const brightness = totalBrightness / (data.length / 4) / 255;
    
    // Calculate contrast (standard deviation of brightness)
    let varianceSum = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const pixelBrightness = (r + g + b) / 3 / 255;
      varianceSum += Math.pow(pixelBrightness - brightness, 2);
    }
    const contrast = Math.sqrt(varianceSum / (data.length / 4));
    
    // Calculate sharpness (edge detection approximation)
    const sharpness = this.calculateSharpness(data, width, height);
    
    // Calculate composition score (rule of thirds approximation)
    const composition = this.calculateComposition(data, width, height);
    
    // Overall quality score
    const qualityScore = (
      brightness * 0.2 +
      Math.min(contrast * 2, 1) * 0.3 +
      sharpness * 0.3 +
      composition * 0.2
    );
    
    return {
      qualityScore: Math.max(0, Math.min(1, qualityScore)),
      brightness: Math.max(0, Math.min(1, brightness)),
      contrast: Math.max(0, Math.min(1, contrast * 2)),
      sharpness: Math.max(0, Math.min(1, sharpness)),
      composition: Math.max(0, Math.min(1, composition))
    };
  }

  private calculateSharpness(data: Uint8ClampedArray, width: number, height: number): number {
    let sharpnessSum = 0;
    let count = 0;
    
    // Simple edge detection using Sobel operator approximation
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        const current = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        const right = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
        const bottom = (data[(y + 1) * width * 4 + x * 4] + 
                      data[(y + 1) * width * 4 + x * 4 + 1] + 
                      data[(y + 1) * width * 4 + x * 4 + 2]) / 3;
        
        const gradientX = Math.abs(right - current);
        const gradientY = Math.abs(bottom - current);
        const gradient = Math.sqrt(gradientX * gradientX + gradientY * gradientY);
        
        sharpnessSum += gradient;
        count++;
      }
    }
    
    return count > 0 ? (sharpnessSum / count) / 255 : 0;
  }

  private calculateComposition(data: Uint8ClampedArray, width: number, height: number): number {
    // Rule of thirds: check if interesting content is near intersection points
    const thirdX1 = Math.floor(width / 3);
    const thirdX2 = Math.floor(2 * width / 3);
    const thirdY1 = Math.floor(height / 3);
    const thirdY2 = Math.floor(2 * height / 3);
    
    const intersections = [
      [thirdX1, thirdY1], [thirdX2, thirdY1],
      [thirdX1, thirdY2], [thirdX2, thirdY2]
    ];
    
    let compositionScore = 0;
    
    intersections.forEach(([x, y]) => {
      const idx = (y * width + x) * 4;
      const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      
      // Check local contrast around intersection
      let localContrast = 0;
      let samples = 0;
      
      for (let dy = -5; dy <= 5; dy++) {
        for (let dx = -5; dx <= 5; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nIdx = (ny * width + nx) * 4;
            const nBrightness = (data[nIdx] + data[nIdx + 1] + data[nIdx + 2]) / 3;
            localContrast += Math.abs(brightness - nBrightness);
            samples++;
          }
        }
      }
      
      if (samples > 0) {
        compositionScore += (localContrast / samples) / 255;
      }
    });
    
    return compositionScore / intersections.length;
  }

  public async detectFaces(imageUrl: string): Promise<{
    faceCount: number;
    mainFaceSize: number;
    facePositions: { x: number; y: number; width: number; height: number }[];
  }> {
    // Simplified face detection simulation
    // In a real implementation, you would use a proper face detection model
    
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        // Simulate face detection results
        const faceCount = Math.random() > 0.3 ? 1 : Math.random() > 0.8 ? 2 : 0;
        const mainFaceSize = faceCount > 0 ? Math.random() * 0.4 + 0.3 : 0; // 30-70% of image
        
        const facePositions = [];
        if (faceCount > 0) {
          facePositions.push({
            x: Math.random() * 0.3 + 0.2, // 20-50% from left
            y: Math.random() * 0.3 + 0.1, // 10-40% from top
            width: mainFaceSize,
            height: mainFaceSize * 1.2 // Faces are typically taller than wide
          });
        }
        
        resolve({ faceCount, mainFaceSize, facePositions });
      };
      
      img.onerror = () => {
        resolve({ faceCount: 0, mainFaceSize: 0, facePositions: [] });
      };
      
      img.src = imageUrl;
    });
  }

  public async analyzeSceneContext(imageUrl: string): Promise<{
    sceneType: string;
    confidence: number;
    tags: string[];
  }> {
    // Simulate scene analysis
    const sceneTypes = [
      { type: 'outdoor', tags: ['nature', 'landscape', 'adventure'] },
      { type: 'indoor', tags: ['home', 'cozy', 'intimate'] },
      { type: 'social', tags: ['party', 'friends', 'social'] },
      { type: 'fitness', tags: ['gym', 'workout', 'healthy'] },
      { type: 'travel', tags: ['vacation', 'explore', 'wanderlust'] },
      { type: 'professional', tags: ['work', 'business', 'career'] },
      { type: 'hobby', tags: ['creative', 'passion', 'skill'] }
    ];
    
    const randomScene = sceneTypes[Math.floor(Math.random() * sceneTypes.length)];
    
    return {
      sceneType: randomScene.type,
      confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
      tags: randomScene.tags
    };
  }

  public calculateOverallAttractiveness(analyses: {
    quality: any;
    faces: any;
    scene: any;
  }): number {
    const qualityWeight = 0.4;
    const faceWeight = 0.4;
    const sceneWeight = 0.2;
    
    const qualityScore = analyses.quality.qualityScore;
    const faceScore = analyses.faces.faceCount > 0 ? 
      Math.min(1, analyses.faces.mainFaceSize * 2) : 0.3;
    const sceneScore = analyses.scene.confidence;
    
    return qualityWeight * qualityScore + 
           faceWeight * faceScore + 
           sceneWeight * sceneScore;
  }
}