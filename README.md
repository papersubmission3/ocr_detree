# OCR-DETree

Detect human- versus AI-generated text from image-based inputs using optical character recognition (OCR) and **DETree**.

This repository contains a Google Colab/Jupyter Notebook implementation that extracts text from screenshots or other image inputs, represents the extracted text with the DETree language model, and compares the resulting representation against reference embeddings to classify the content as **AI-generated** or **human-written**.

> **Status:** Research/prototype implementation. Predictions should not be treated as definitive evidence of authorship.

## Repository structure

```text
.
├── Code/
│   └── DETree_NeurIPS2025_Complete_MT1_2.ipynb
└── Dataset/
    ├── Essay/
    ├── arxiv/
    ├── news/
    └── writing/
```

The main notebook includes:

- GPU availability checks
- Dependency installation
- DETree model loading
- Reference embedding download and validation
- FAISS nearest-neighbor search
- OCR-based screenshot detection
- Supervised detection and robustness experiments
- t-SNE and ablation visualizations
- A Gradio interactive demo

## Quick start

The notebook is designed to run in **Google Colab**. Open [`Code/DETree_NeurIPS2025_Complete_MT1_2.ipynb`](Code/DETree_NeurIPS2025_Complete_MT1_2.ipynb) and run the cells from top to bottom.

### Recommended runtime

A CUDA-enabled runtime is strongly recommended:

1. In Colab, select **Runtime → Change runtime type**.
2. Choose a **T4 GPU** or another compatible GPU.
3. Open the notebook and execute the GPU check before continuing.

The notebook can use CPU as a fallback for some operations, but model loading and inference may be substantially slower and the complete workflow may require more memory than a typical CPU runtime provides.

### External resources

During execution, the notebook downloads resources from external repositories, including:

- The DETree model from Hugging Face: [`heyongxin233/DETree`](https://huggingface.co/heyongxin233/DETree)
- Reference datasets and embeddings from Hugging Face: [`heyongxin233/RealBench`](https://huggingface.co/datasets/heyongxin233/RealBench)
- DETree source code and requirements from the upstream implementation referenced in the notebook

A Hugging Face account/token may be requested by the notebook. Public resources may be accessible without authentication, but configuring `HF_TOKEN` in Colab can improve reliability and avoid rate limits.

## Installation

The notebook installs its Python dependencies automatically. The primary packages include:

- PyTorch
- Transformers
- FAISS
- scikit-learn
- pandas
- NumPy
- Matplotlib and Seaborn
- Pillow
- `pytesseract`
- Gradio

If you adapt the notebook for a local environment, install the dependencies in its setup cells and ensure that the Tesseract OCR executable is also installed and available on your `PATH`.

## How it works

At a high level, the pipeline is:

1. **Input:** Provide an image containing text, such as a screenshot or scanned document.
2. **OCR:** Extract the visible text from the image.
3. **Representation:** Encode the extracted text using DETree.
4. **Retrieval:** Compare the representation with normalized reference embeddings using FAISS inner-product search.
5. **Classification:** Aggregate the nearest-neighbor labels to produce an AI-versus-human prediction.
6. **Visualization/demo:** Inspect evaluation results or use the Gradio interface for interactive inference.

The reference embedding databases use the labels `0 = AI` and `1 = human` as configured in the notebook. Verify the label mapping before replacing or combining embedding databases.

## Data

The `Dataset/` directory organizes project data by source or writing domain, including essays, arXiv content, news, and general writing. The notebook also retrieves additional benchmark data and embedding files from the RealBench dataset repository at runtime.

Large model and embedding artifacts are intentionally downloaded during execution rather than duplicated in this repository.

## Reproducibility notes

- Run cells in order because later sections depend on downloaded models, datasets, and initialized indexes.
- Use the same model, embedding layer, class ordering, and normalization settings when reproducing results.
- Confirm that embedding dimensions and label mappings match before building a combined FAISS index.
- Network access is required for the first run.
- Results can vary with OCR quality, image resolution, font/layout, preprocessing, and the selected reference data.

## Responsible use

AI-text detection is probabilistic and can produce false positives and false negatives. OCR introduces additional errors, especially for low-resolution, stylized, handwritten, or multilingual inputs. Do not use this project as the sole basis for academic, employment, legal, disciplinary, or other high-impact decisions. Obtain consent where appropriate and review predictions alongside the original image and contextual evidence.

## Acknowledgements

This project builds on the DETree model and the RealBench resources referenced in the notebook. Please consult the upstream repositories and associated papers for the original model, dataset, and licensing information before redistributing or using them.

## Citation

If you use this implementation in academic work, cite the DETree paper and the upstream model/dataset releases referenced by the notebook. Add the final bibliographic information here once the project’s preferred citation format has been confirmed.
