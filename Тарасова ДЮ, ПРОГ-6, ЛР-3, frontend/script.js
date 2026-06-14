// ========== КОНФИГУРАЦИЯ ==========
const USE_MOCK = true;   // true = заглушки, false = реальный бэкенд
const API_BASE = 'http://localhost:8000';

// ========== ТЕМА (светлая/тёмная) ==========
function initTheme() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'light') {
    document.body.classList.remove('dark-theme');
    document.querySelector('.theme-toggle span:last-child').textContent = 'Светлая тема';
    document.querySelector('.theme-icon').textContent = '☀️';
  } else {
    document.body.classList.add('dark-theme');
    document.querySelector('.theme-toggle span:last-child').textContent = 'Тёмная тема';
    document.querySelector('.theme-icon').textContent = '🌙';
  }
}

function toggleTheme() {
  if (document.body.classList.contains('dark-theme')) {
    document.body.classList.remove('dark-theme');
    localStorage.setItem('theme', 'light');
    document.querySelector('.theme-toggle span:last-child').textContent = 'Светлая тема';
    document.querySelector('.theme-icon').textContent = '☀️';
  } else {
    document.body.classList.add('dark-theme');
    localStorage.setItem('theme', 'dark');
    document.querySelector('.theme-toggle span:last-child').textContent = 'Тёмная тема';
    document.querySelector('.theme-icon').textContent = '🌙';
  }
}

// ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
function showLoading() {
  document.getElementById('loadingOverlay').style.display = 'flex';
}
function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}
function showError(msg) {
  alert('Ошибка: ' + msg);
}

function getFormData() {
  return {
    person_age: parseFloat(document.getElementById('person_age').value),
    person_gender: document.getElementById('person_gender').value,
    person_education: document.getElementById('person_education').value,
    person_income: parseFloat(document.getElementById('person_income').value),
    person_emp_exp: parseInt(document.getElementById('person_emp_exp').value),
    person_home_ownership: document.getElementById('person_home_ownership').value,
    loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
    loan_intent: document.getElementById('loan_intent').value,
    loan_int_rate: parseFloat(document.getElementById('loan_int_rate').value),
    loan_percent_income: parseFloat(document.getElementById('loan_percent_income').value),
    cb_person_cred_hist_length: parseFloat(document.getElementById('cb_person_cred_hist_length').value),
    credit_score: parseInt(document.getElementById('credit_score').value),
    previous_loan_defaults_on_file: document.getElementById('previous_loan_defaults_on_file').value
  };
}

function displaySinglePrediction(response) {
  let status = response.loan_status || response.prediction;
  if (!status) status = response[0]?.loan_status;
  const resultDiv = document.getElementById('singleResult');
  const badgeDiv = document.querySelector('.result-badge');
  if (status === 'approved' || status === 1 || status === '1') {
    badgeDiv.textContent = 'Одобрено';
    badgeDiv.className = 'result-badge approved';
  } else {
    badgeDiv.textContent = 'Отказано';
    badgeDiv.className = 'result-badge rejected';
  }
  resultDiv.style.display = 'block';
}

// ========== MOCK ==========
async function mockPredict(data) {
  await new Promise(r => setTimeout(r, 900));
  const approved = (data.person_income > 50000 && data.credit_score > 650);
  return { loan_status: approved ? 'approved' : 'rejected' };
}

async function mockPredictFromCsv(file) {
  await new Promise(r => setTimeout(r, 1300));
  const fakeRoc = (0.75 + Math.random() * 0.21).toFixed(3);
  const text = await file.text();
  const rows = text.split('\n').slice(0, 12);
  const headers = rows[0].split(',');
  const dataRows = rows.slice(1).filter(r => r.trim() !== '').map(row => {
    const cols = row.split(',');
    const obj = {};
    headers.forEach((h, idx) => obj[h.trim()] = cols[idx] || '');
    obj.predicted_loan_status = Math.random() > 0.3 ? 'approved' : 'rejected';
    return obj;
  });
  return { roc_auc: parseFloat(fakeRoc), data: dataRows };
}

// ========== REAL API ==========
async function realPredict(data) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const result = await response.json();
  return Array.isArray(result) ? result[0] : result;
}

async function realPredictFromCsv(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE}/predict-from-csv`, {
    method: 'POST',
    body: formData
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json();
}

// ========== ОБЁРТКИ ==========
async function predictLoan(data) {
  return USE_MOCK ? mockPredict(data) : realPredict(data);
}
async function predictCsv(file) {
  return USE_MOCK ? mockPredictFromCsv(file) : realPredictFromCsv(file);
}

function renderCsvTable(roc, dataArray) {
  document.getElementById('csvResult').style.display = 'block';
  document.getElementById('rocValue').innerText = roc;
  const tbody = document.querySelector('#resultTable tbody');
  tbody.innerHTML = '';
  if (!dataArray || dataArray.length === 0) return;
  const first = dataArray[0];
  const keys = Object.keys(first);
  const thead = document.querySelector('#resultTable thead');
  thead.innerHTML = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
  for (let row of dataArray) {
    const tr = document.createElement('tr');
    keys.forEach(k => {
      const td = document.createElement('td');
      td.textContent = row[k] !== undefined ? row[k] : '';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

// ========== EVENT HANDLERS ==========
document.getElementById('loanForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  showLoading();
  try {
    const data = getFormData();
    const result = await predictLoan(data);
    displaySinglePrediction(result);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
});

document.getElementById('uploadCsvBtn').addEventListener('click', async () => {
  const file = document.getElementById('csvFile').files[0];
  if (!file) {
    showError('Выберите CSV-файл');
    return;
  }
  showLoading();
  try {
    const resp = await predictCsv(file);
    const roc = resp.roc_auc !== undefined ? resp.roc_auc : '—';
    const dataRows = resp.data || resp.dataset || [];
    renderCsvTable(roc, dataRows);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
});

// Инициализация темы
document.getElementById('themeToggle').addEventListener('click', toggleTheme);
initTheme();