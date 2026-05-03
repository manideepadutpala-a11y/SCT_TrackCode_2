document.addEventListener("DOMContentLoaded", function () {
  const tabButtons = document.querySelectorAll("[data-tab]");
  const panels = document.querySelectorAll("[data-panel]");

  function setActiveTab(name) {
    tabButtons.forEach((button) => {
      const active = button.dataset.tab === name;
      button.classList.toggle("active", active);
    });
    panels.forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.panel !== name);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      setActiveTab(button.dataset.tab);
    });
  });

  function buildCharts() {
    if (!window.dashboardData || typeof window.Chart === "undefined") {
      return;
    }

    const { genderLabels, genderCounts, spendLabels, spendValues, clusterLabels, clusterCounts, clusterIncome, scatterPoints } = window.dashboardData;

    const genderPie = document.getElementById("genderPie");
    const spendBar = document.getElementById("spendBar");
    const clusterBar = document.getElementById("clusterBar");
    const scatterChart = document.getElementById("scatterChart");

    if (genderPie) {
      new Chart(genderPie.getContext("2d"), {
        type: "pie",
        data: {
          labels: genderLabels,
          datasets: [{
            data: genderCounts,
            backgroundColor: ["#f49ac2", "#7ec0ee"],
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });
    }

    if (spendBar) {
      new Chart(spendBar.getContext("2d"), {
        type: "bar",
        data: {
          labels: spendLabels,
          datasets: [{
            label: "Avg Spending Score",
            data: spendValues,
            backgroundColor: ["#f49ac2", "#7ec0ee"],
            borderRadius: 10,
          }],
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: true, max: 100 },
          },
        },
      });
    }

    if (clusterBar) {
      new Chart(clusterBar.getContext("2d"), {
        type: "bar",
        data: {
          labels: clusterLabels,
          datasets: [{
            label: "Cluster Size",
            data: clusterCounts,
            backgroundColor: ["#f9d25c", "#8dc63f", "#e94f37", "#7f5f9b", "#1f1f1f"],
            borderRadius: 10,
          }],
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: true },
          },
        },
      });
    }

    if (scatterChart) {
      const clusterColors = ["#f49ac2", "#7ec0ee", "#f9d25c", "#8dc63f", "#1f1f1f"];
      new Chart(scatterChart.getContext("2d"), {
        type: "scatter",
        data: {
          datasets: clusterLabels.map((label, idx) => ({
            label,
            data: scatterPoints.filter((point) => point.cluster === idx),
            backgroundColor: clusterColors[idx % clusterColors.length],
            pointRadius: 6,
          })),
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: "Annual Income (k$)" } },
            y: { title: { display: true, text: "Spending Score" }, min: 0, max: 100 },
          },
        },
      });
    }
  }

  buildCharts();
  setActiveTab("summary");
});
