import streamlit as st
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import os

# 1. Configuração da Página
st.set_page_config(page_title="IA Orca Mobile", page_icon="🐋")

st.title("🐋 Orca Mini 3B (Sem Token)")
st.caption("Método de download direto. Inteligência local sem login.")

# 2. Função de Download Blindada (Resolve o erro 401)
@st.cache_resource
def download_and_load_model():
    model_id = "TheBloke/Orca-Mini-3B-GGUF"
    filename = "orca-mini-3b.q4_0.gguf"
    
    status_placeholder = st.empty()
    status_placeholder.info(f"📥 Baixando modelo {model_id}... Isso garante que funcione sem token.")
    
    try:
        # Aqui forçamos o download do arquivo público para uma pasta local
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=".",  # Baixa na pasta raiz do app
            local_dir_use_symlinks=False
        )
        status_placeholder.success("✅ Download concluído! Iniciando o cérebro...")
        
        # Agora carregamos o arquivo LOCALMENTE (Zero chance de erro 401)
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            gpu_layers=0,
            context_length=1024
        )
        status_placeholder.empty() # Limpa a mensagem
        return llm
        
    except Exception as e:
        status_placeholder.error(f"Erro no download ou carregamento: {e}")
        st.stop()

# Carrega a IA
llm = download_and_load_model()

# 3. Histórico
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou o Orca. Consegui furar o bloqueio. Vamos conversar?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. Chat
if prompt := st.chat_input("Diga algo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""
        
        # Prompt simples
        full_prompt = f"### User:\n{prompt}\n\n### Response:\n"
        
        try:
            for text in llm(full_prompt, stream=True, max_new_tokens=512, temperature=0.7):
                response_text += text
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error("Memória cheia. Tente recarregar.")

st.sidebar.success("Status: Hack de Download Ativo")
