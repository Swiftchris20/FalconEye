// File: PyTorchClassifier3.kt
package com.developer27.falconeye.inference

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import kotlin.math.exp

/**
 * A PyTorch classifier that slices a 4-logit model into 3 labels.
 */
class PyTorchClassifier private constructor(
    private val module: Module,
    val inputWidth: Int = 224,
    val inputHeight: Int = 224
) : AutoCloseable {

    companion object {
        @Volatile private var cachedModule: Module? = null

        fun fromAsset(
            context: Context,
            modelAsset: String = "resnet50_speed.pt",
            inputWidth: Int = 224,
            inputHeight: Int = 224
        ): PyTorchClassifier {
            val mod = cachedModule ?: synchronized(this) {
                cachedModule ?: PyTorchModuleLoader
                    .loadModule(context, modelAsset)
                    .also { cachedModule = it }
            }
            return PyTorchClassifier(mod, inputWidth, inputHeight)
        }
    }

    /** Types of speed */
    private val labels = listOf("Speed Type 1", "Speed Type 2", "Speed Type 3", "Speed Type 4")

    // ImageNet normalization constants
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Reusable softmax buffer of size 3 */
    private val softmaxBuffer = FloatArray(labels.size)

    /**
     * Classifies a single bitmap by:
     * 1. Scaling & normalizing the bitmap to the model’s input dimensions
     * 2. Running the full N‑logit model to obtain raw outputs
     * 3. Applying softmax across all logits to compute class probabilities
     * 4. Selecting the label with the highest probability
     */
    fun classifyLine(bitmap: Bitmap): Pair<String, FloatArray> {
        // 1) Pre‑process
        val scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(scaled, mean, std)

        // 2) Run the model and get *all* logits
        val logits = module.forward(IValue.from(tensor))
            .toTensor()
            .dataAsFloatArray

        // 3) Softmax over *all* logits
        val probs = softmax(logits)

        // 4) Find the best index
        val best = probs.indices.maxByOrNull { probs[it] } ?: 0

        // 5) Return the corresponding label + full probability array
        return labels[best] to probs
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = logits[0]
        for (i in 1 until logits.size) if (logits[i] > maxLogit) maxLogit = logits[i]
        var sum = 0f
        for (i in logits.indices) {
            val e = exp(logits[i] - maxLogit)
            softmaxBuffer[i] = e
            sum += e
        }
        for (i in softmaxBuffer.indices) {
            softmaxBuffer[i] = softmaxBuffer[i] / sum
        }
        return softmaxBuffer
    }

    override fun close() {
        module.destroy()
    }
}