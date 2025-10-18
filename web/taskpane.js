let SERVER = "https://localhost:8000";

Office.initialize = () => {
  bindUI();
  ping();
};

function el(id){ return document.getElementById(id); }
function escapeHtml(s){ return (s||"").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function addMsg(role, text, citations=[]) {
  const box = document.createElement('div');
  box.className = `message ${role}`;
  box.innerHTML = `
    <div class="label">${role === 'user' ? 'Bạn' : 'Assistant'}</div>
    <div class="text">${escapeHtml(text).replace(/\n/g,'<br/>')}</div>
    ${citations && citations.length ? `
      <div class="citations">
        ${citations.map(c => `<div class="citation"><strong>${escapeHtml(c.source||'src')}</strong> — ${escapeHtml(c.snippet||'')}</div>`).join('')}
      </div>` : ''
    }
  `;
  el('chat').appendChild(box);
  el('chat').scrollTop = el('chat').scrollHeight;
}

async function getSelectionOrDocument(scope){
  if(scope === 'document'){
    return Word.run(async (ctx) => {
      const body = ctx.document.body;
      body.load('text');
      await ctx.sync();
      return body.text || "";
    });
  }
  return new Promise((resolve) => {
    Office.context.document.getSelectedDataAsync(Office.CoercionType.Text, (res) => {
      resolve(res.status === Office.AsyncResultStatus.Succeeded ? (res.value || "") : "");
    });
  });
}

function bindUI(){
  // Tabs
  [...document.querySelectorAll('.tab')].forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    });
  });

  // Chat submit
  el('composer').addEventListener('submit', async (e)=>{
    e.preventDefault();
    const q = el('question').value.trim();
    if(!q) return;
    const scope = document.querySelector('input[name="scope"]:checked').value;
    const useKB = el('use-kb').checked;
    const withCite = el('with-citations').checked;

    addMsg('user', q);
    el('question').value = '';

    const ctxText = await getSelectionOrDocument(scope);
    try{
      const resp = await fetch(`${SERVER}/ask`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          question: q,
          context: ctxText,
          options: { useKnowledgeBase: useKB, withCitations: withCite }
        })
      });
      const data = await resp.json();
      addMsg('assistant', data.answer || '(no answer)', data.citations||[]);
    }catch(err){
      addMsg('assistant', '❌ Lỗi gọi server: ' + err.message);
    }
  });

  // KB stats
  el('refresh-stats').addEventListener('click', loadStats);
  loadStats();

  // Index current doc / selection
  el('index-current-doc').addEventListener('click', async ()=>{
    await indexFromWord('document');
  });
  el('index-selection').addEventListener('click', async ()=>{
    await indexFromWord('selection');
  });

  // KB search
  el('kb-search-btn').addEventListener('click', async ()=>{
    const q = el('kb-search-query').value.trim();
    if(!q) return;
    const r = await fetch(`${SERVER}/search_knowledge_base`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,top_k:5})
    });
    const data = await r.json();
    const box = el('kb-search-results');
    if(!data.success){ box.innerHTML = `<p>Lỗi: ${escapeHtml(JSON.stringify(data))}</p>`; return; }
    box.innerHTML = data.matches.map(m=>`
      <div class="search-result">
        <div class="score">score: ${m.score.toFixed(3)} – doc: ${escapeHtml(m.metadata.doc_id||'')}</div>
        <div class="text">${escapeHtml(m.text)}</div>
      </div>
    `).join('');
  });
}

async function ping(){
  try{
    const r = await fetch(`${SERVER}/ping`);
    const j = await r.json();
    // optional: show a toast/status
  }catch(e){
    addMsg('assistant','⚠️ Không thể ping server. Hãy kiểm tra HTTPS/certificate/port.');
  }
}

async function indexFromWord(scope){
  const title = el('doc-title').value.trim() || 'untitled';
  const typ = el('doc-type').value || 'other';
  const date = el('doc-date').value || '';
  const content = await getSelectionOrDocument(scope);

  const st = el('index-status');
  if(!content){ st.className='status-message error'; st.textContent='Không có nội dung để index.'; return; }

  try{
    const r = await fetch(`${SERVER}/index_document`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        content,
        metadata: { title, type: typ, date_issued: date }
      })
    });
    const data = await r.json();
    if(data.success){
      st.className='status-message success';
      st.textContent=`Đã index ${data.chunks_indexed} chunks (doc_id=${data.doc_id}).`;
    }else{
      st.className='status-message error';
      st.textContent=`Index thất bại: ${JSON.stringify(data)}`;
    }
  }catch(e){
    st.className='status-message error';
    st.textContent=`Lỗi gọi API: ${e.message}`;
  }
}

