// FreshHarvest AI - Admin Analytics & Chart.js Visualizations

document.addEventListener('DOMContentLoaded', () => {
    initFreshnessDonut();
    initFruitBreakdownBar();
    initModelBenchmarkChart();
    initConfusionMatrixTabs();
});

function initFreshnessDonut() {
    const ctx = document.getElementById('chart-freshness-donut');
    if (!ctx || typeof analyticsData === 'undefined') return;

    const fresh = analyticsData.fresh_count || 0;
    const spoiled = analyticsData.spoiled_count || 0;
    const total = fresh + spoiled;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Commercial Fresh', 'Spoiled / Degraded'],
            datasets: [{
                data: total === 0 ? [1, 1] : [fresh, spoiled],
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 2,
                borderColor: '#ffffff',
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 14,
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: 600 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw || 0;
                            const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 50;
                            return ` ${context.label}: ${val} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: '72%'
        }
    });
}

function initFruitBreakdownBar() {
    const ctx = document.getElementById('chart-fruit-bar');
    if (!ctx || typeof analyticsData === 'undefined') return;

    const breakdown = analyticsData.fruit_breakdown || [];
    let labels = [];
    let freshData = [];
    let spoiledData = [];

    if (breakdown.length === 0) {
        labels = ['Apple', 'Banana', 'Orange', 'Mango', 'Strawberry', 'Tomato', 'Lemon'];
        freshData = [15, 15, 15, 15, 15, 15, 15];
        spoiledData = [15, 15, 15, 15, 15, 15, 15];
    } else {
        breakdown.forEach(item => {
            labels.push(item.fruit_type);
            freshData.push(item.fresh || 0);
            spoiledData.push(item.spoiled || 0);
        });
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Fresh',
                    data: freshData,
                    backgroundColor: '#10b981',
                    borderRadius: 6
                },
                {
                    label: 'Spoiled',
                    data: spoiledData,
                    backgroundColor: '#ef4444',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Plus Jakarta Sans', weight: 600 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#f1f5f9' },
                    ticks: { stepSize: 5 }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { boxWidth: 12, font: { weight: 600 } }
                }
            }
        }
    });
}

function initModelBenchmarkChart() {
    const ctx = document.getElementById('chart-model-benchmark');
    if (!ctx || typeof modelMetricsData === 'undefined') return;

    let labels = [];
    let accuracies = [];
    let precisions = [];
    let recalls = [];
    let f1Scores = [];

    modelMetricsData.forEach(m => {
        labels.push(m.model_name);
        accuracies.push(parseFloat((m.accuracy * 100).toFixed(1)));
        precisions.push(parseFloat((m.precision * 100).toFixed(1)));
        recalls.push(parseFloat((m.recall * 100).toFixed(1)));
        f1Scores.push(parseFloat((m.f1_score * 100).toFixed(1)));
    });

    if (labels.length === 0) {
        labels = ['EfficientNet-B0', 'ResNet-50', 'MobileNet-V2'];
        accuracies = [25.0, 6.25, 9.38];
        precisions = [11.56, 0.39, 0.88];
        recalls = [25.0, 6.25, 9.38];
        f1Scores = [15.22, 0.74, 1.61];
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Accuracy %', data: accuracies, backgroundColor: '#10b981', borderRadius: 4 },
                { label: 'Precision %', data: precisions, backgroundColor: '#3b82f6', borderRadius: 4 },
                { label: 'Recall %', data: recalls, backgroundColor: '#f59e0b', borderRadius: 4 },
                { label: 'F1-Score %', data: f1Scores, backgroundColor: '#8b5cf6', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: '#f1f5f9' },
                    ticks: {
                        callback: function(val) { return val + '%'; }
                    }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12, font: { weight: 600 } }
                }
            }
        }
    });
}

function initConfusionMatrixTabs() {
    const buttons = document.querySelectorAll('#cm-model-toggle button');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const targetModel = btn.getAttribute('data-model');
            const panes = document.querySelectorAll('.cm-pane');
            panes.forEach(p => {
                if (p.id === `pane-${targetModel}`) {
                    p.classList.remove('d-none');
                } else {
                    p.classList.add('d-none');
                }
            });
        });
    });
}

