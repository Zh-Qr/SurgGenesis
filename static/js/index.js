const videos = {
  case_03: {
    label: "Case 03",
    sources: {
      condition: {
        file: "assets/demo/videos/case_03/condition.mp4",
        label: "Observed context",
        description: "Observed context supplied to the world model before future prediction."
      },
      base: {
        file: "assets/demo/videos/case_03/base_generated.mp4",
        label: "Native prediction",
        description: "Future prediction from the native video foundation model."
      },
      stage1: {
        file: "assets/demo/videos/case_03/stage1_generated.mp4",
        label: "Stage 1 prediction",
        description: "Domain-transferred prediction after the first progressive adaptation stage."
      },
      stage2: {
        file: "assets/demo/videos/case_03/stage2_generated.mp4",
        label: "Stage 2 prediction",
        description: "Triplet-conditioned prediction after the behavior-understanding stage."
      },
      stage3: {
        file: "assets/demo/videos/case_03/stage3_generated.mp4",
        label: "Stage 3 prediction",
        description: "Trajectory-conditioned future prediction from SurgGenesis."
      },
      ground_truth_future: {
        file: "assets/demo/videos/case_03/ground_truth_future.mp4",
        label: "Ground-truth future",
        description: "Held-out future video from the surgical case."
      }
    }
  },
  case_05: {
    label: "Case 05",
    sources: {
      condition: {
        file: "assets/demo/videos/case_05/condition.mp4",
        label: "Observed context",
        description: "Observed context supplied to the world model before future prediction."
      },
      base: {
        file: "assets/demo/videos/case_05/base_generated.mp4",
        label: "Native prediction",
        description: "Future prediction from the native video foundation model."
      },
      stage1: {
        file: "assets/demo/videos/case_05/stage1_generated.mp4",
        label: "Stage 1 prediction",
        description: "Domain-transferred prediction after the first progressive adaptation stage."
      },
      stage2: {
        file: "assets/demo/videos/case_05/stage2_generated.mp4",
        label: "Stage 2 prediction",
        description: "Triplet-conditioned prediction after the behavior-understanding stage."
      },
      stage3: {
        file: "assets/demo/videos/case_05/stage3_generated.mp4",
        label: "Stage 3 prediction",
        description: "Trajectory-conditioned future prediction from SurgGenesis."
      },
      ground_truth_future: {
        file: "assets/demo/videos/case_05/ground_truth_future.mp4",
        label: "Ground-truth future",
        description: "Held-out future video from the surgical case."
      }
    }
  }
};

const state = {
  caseId: "case_03",
  stage: "condition"
};

const comparisonVideo = document.querySelector("#comparison-video");
const comparisonLabel = document.querySelector("#comparison-label");
const comparisonDescription = document.querySelector("#comparison-description");

function updateComparison() {
  const selected = videos[state.caseId].sources[state.stage];
  const shouldPlay = !comparisonVideo.paused;

  comparisonVideo.src = selected.file;
  comparisonVideo.load();
  comparisonLabel.textContent = `${videos[state.caseId].label} · ${selected.label}`;
  comparisonDescription.textContent = selected.description;

  if (shouldPlay) {
    comparisonVideo.play().catch(() => {});
  }
}

function setSelectedButton(selector, activeValue, datasetKey) {
  document.querySelectorAll(selector).forEach((button) => {
    const isActive = button.dataset[datasetKey] === activeValue;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
}

document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", () => {
    state.caseId = button.dataset.case;
    setSelectedButton("[data-case]", state.caseId, "case");
    updateComparison();
  });
});

document.querySelectorAll("[data-stage]").forEach((button) => {
  button.addEventListener("click", () => {
    state.stage = button.dataset.stage;
    setSelectedButton("[data-stage]", state.stage, "stage");
    updateComparison();
  });
});

document.querySelector("#replay-video").addEventListener("click", () => {
  comparisonVideo.currentTime = 0;
  comparisonVideo.play().catch(() => {});
});

document.querySelector("#copy-citation").addEventListener("click", async (event) => {
  const citation = document.querySelector("#citation-text").innerText;
  const button = event.currentTarget;

  try {
    await navigator.clipboard.writeText(citation);
    button.setAttribute("title", "Citation copied");
    button.setAttribute("aria-label", "Citation copied");
    setTimeout(() => {
      button.setAttribute("title", "Copy BibTeX citation");
      button.setAttribute("aria-label", "Copy BibTeX citation");
    }, 1800);
  } catch {
    window.getSelection().selectAllChildren(document.querySelector("#citation-text"));
  }
});

window.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
