const cases = {
  case_03: {
    observation: "assets/demo/videos/case_03/condition.mp4",
    stage1: "assets/demo/videos/case_03/stage1_generated.mp4",
    stage2: "assets/demo/videos/case_03/stage2_generated.mp4",
    stage3: "assets/demo/videos/case_03/stage3_generated.mp4"
  },
  case_05: {
    observation: "assets/demo/videos/case_05/condition.mp4",
    stage1: "assets/demo/videos/case_05/stage1_generated.mp4",
    stage2: "assets/demo/videos/case_05/stage2_generated.mp4",
    stage3: "assets/demo/videos/case_05/stage3_generated.mp4"
  }
};

function updateCase(caseId) {
  const selectedCase = cases[caseId];
  const observation = document.querySelector("#observation-video");

  observation.src = selectedCase.observation;
  observation.load();
  observation.play().catch(() => {});

  document.querySelectorAll("[data-stage-video]").forEach((video) => {
    video.src = selectedCase[video.dataset.stageVideo];
    video.load();
    video.play().catch(() => {});
  });
}

document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", () => {
    const isSelected = button.classList.contains("is-active");
    if (isSelected) return;

    document.querySelectorAll("[data-case]").forEach((tab) => {
      const isActive = tab === button;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });

    updateCase(button.dataset.case);
  });
});

if (window.lucide) {
  window.lucide.createIcons();
}
