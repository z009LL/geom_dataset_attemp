using Lux
using Random
using LinearAlgebra

# Attention Multi-Head
struct MultiHeadAttention
    num_heads::Int
    head_size::Int
    Wq::Dense
    Wk::Dense
    Wv::Dense
    Wo::Dense
end

function MultiHeadAttention(input_dim::Int, num_heads::Int)
    head_size = div(input_dim, num_heads)
    return MultiHeadAttention(
        num_heads,
        head_size,
        Dense(input_dim => input_dim),
        Dense(input_dim => input_dim),
        Dense(input_dim => input_dim),
        Dense(input_dim => input_dim)
    )
end

function (mha::MultiHeadAttention)(x, ps, st)
    batch_size, seq_len, _ = size(x)
    
    q, stq = mha.Wq(x, ps.Wq, st.Wq)
    k, stk = mha.Wk(x, ps.Wk, st.Wk)
    v, stv = mha.Wv(x, ps.Wv, st.Wv)
    
    # Reshape manteniendo el batch_size como primera dimensión
    q = reshape(q, batch_size, seq_len, mha.num_heads, :)
    k = reshape(k, batch_size, seq_len, mha.num_heads, :)
    v = reshape(v, batch_size, seq_len, mha.num_heads, :)
    
    # Transponer para alinear las dimensiones correctamente
    q = permutedims(q, (1, 3, 2, 4))  # (batch, heads, seq, depth)
    k = permutedims(k, (1, 3, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))
    
    # Realizar la multiplicación por lotes
    scores = batched_mul(q, permutedims(k, (1, 2, 4, 3))) ./ sqrt(Float32(mha.head_size))
    attn_weights = softmax(scores, dims=4)
    attn_output = batched_mul(attn_weights, v)
    
    # Volver a la forma original
    attn_output = permutedims(attn_output, (1, 3, 2, 4))
    attn_output = reshape(attn_output, batch_size, seq_len, :)
    
    output, sto = mha.Wo(attn_output, ps.Wo, st.Wo)
    
    new_st = (Wq=stq, Wk=stk, Wv=stv, Wo=sto)
    return output, new_st
end
function LuxCore.initialparameters(rng::AbstractRNG, mha::MultiHeadAttention)
    return (
        Wq = LuxCore.initialparameters(rng, mha.Wq),
        Wk = LuxCore.initialparameters(rng, mha.Wk),
        Wv = LuxCore.initialparameters(rng, mha.Wv),
        Wo = LuxCore.initialparameters(rng, mha.Wo)
    )
end

function LuxCore.initialstates(rng::AbstractRNG, mha::MultiHeadAttention)
    return (
        Wq = LuxCore.initialstates(rng, mha.Wq),
        Wk = LuxCore.initialstates(rng, mha.Wk),
        Wv = LuxCore.initialstates(rng, mha.Wv),
        Wo = LuxCore.initialstates(rng, mha.Wo)
    )
end

# Feed-Forward Layer
struct FeedForward
    dense1::Dense
    dense2::Dense
end

function FeedForward(input_dim::Int, hidden_dim::Int)
    return FeedForward(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => input_dim)
    )
end

function (ff::FeedForward)(x, ps, st)
    x, st1 = ff.dense1(x, ps.dense1, st.dense1)
    x, st2 = ff.dense2(x, ps.dense2, st.dense2)
    return x, (dense1=st1, dense2=st2)
end

function LuxCore.initialparameters(rng::AbstractRNG, ff::FeedForward)
    return (
        dense1 = LuxCore.initialparameters(rng, ff.dense1),
        dense2 = LuxCore.initialparameters(rng, ff.dense2)
    )
end
function LuxCore.initialstates(rng::AbstractRNG, ff::FeedForward)
    return (
        dense1 = LuxCore.initialstates(rng, ff.dense1),
        dense2 = LuxCore.initialstates(rng, ff.dense2)
    )
end



struct EncoderLayer
    attention::MultiHeadAttention
    norm1::LayerNorm
    feedforward::FeedForward
    norm2::LayerNorm
end

function EncoderLayer(input_dim::Int, num_heads::Int, ff_dim::Int)
    return EncoderLayer(
        MultiHeadAttention(input_dim, num_heads),
        LayerNorm((input_dim,)),
        FeedForward(input_dim, ff_dim),
        LayerNorm((input_dim,))
    )
end

function (el::EncoderLayer)(x, ps, st)
    attn_output, st_attn = el.attention(x, ps.attention, st.attention)
    x = x + attn_output
    x, st_norm1 = el.norm1(x, ps.norm1, st.norm1)
    
    ff_output, st_ff = el.feedforward(x, ps.feedforward, st.feedforward)
    x = x + ff_output
    x, st_norm2 = el.norm2(x, ps.norm2, st.norm2)
    
    new_st = (attention=st_attn, norm1=st_norm1, feedforward=st_ff, norm2=st_norm2)
    return x, new_st
end

# Añadir método de inicialización para EncoderLayer
function LuxCore.initialparameters(rng::AbstractRNG, el::EncoderLayer)
    return (
        attention = LuxCore.initialparameters(rng, el.attention),
        norm1 = LuxCore.initialparameters(rng, el.norm1),
        feedforward = LuxCore.initialparameters(rng, el.feedforward),
        norm2 = LuxCore.initialparameters(rng, el.norm2)
    )
end
function LuxCore.initialstates(rng::AbstractRNG, el::EncoderLayer)
    return (
        attention = LuxCore.initialstates(rng, el.attention),
        norm1 = LuxCore.initialstates(rng, el.norm1),
        feedforward = LuxCore.initialstates(rng, el.feedforward),
        norm2 = LuxCore.initialstates(rng, el.norm2)
    )
end

# Transformer Modificado
struct Transformer
    embed::Dense
    encoder_layers::Vector{EncoderLayer}
    output_layer::Dense
end

function Transformer(vocab_size::Int, d_model::Int, num_heads::Int, num_layers::Int, ff_dim::Int)
    encoder_layers = [EncoderLayer(d_model, num_heads, ff_dim) for _ in 1:num_layers]
    return Transformer(
        
        Dense(vocab_size => d_model),
        encoder_layers,
        Dense(d_model => vocab_size)
    )
end

function (t::Transformer)(x, ps, st)

    x = permutedims(x, (2, 1, 3))
    x, st_embed = t.embed(x, ps.embed, st.embed)
    
    st_encoder = []
    for (i, layer) in enumerate(t.encoder_layers)
        x, st_layer = layer(x, ps.encoder_layers[i], st.encoder_layers[i])
        push!(st_encoder, st_layer)
    end
    
    x, st_output = t.output_layer(x, ps.output_layer, st.output_layer)
    new_st = (embed=st_embed, encoder_layers=tuple(st_encoder...), output_layer=st_output)
    return x, new_st
end



# Añadir método de inicialización para Transformer
function LuxCore.initialparameters(rng::AbstractRNG, t::Transformer)
    return (
        embed = LuxCore.initialparameters(rng, t.embed),
        encoder_layers = tuple([LuxCore.initialparameters(rng, layer) for layer in t.encoder_layers]...),
        output_layer = LuxCore.initialparameters(rng, t.output_layer)
    )
end
function LuxCore.initialstates(rng::AbstractRNG, t::Transformer)
    return (
        embed = LuxCore.initialstates(rng, t.embed),
        encoder_layers = tuple([LuxCore.initialstates(rng, layer) for layer in t.encoder_layers]...),
        output_layer = LuxCore.initialstates(rng, t.output_layer)
    )
end
# Ejemplo de uso
rng = Random.default_rng()
vocab_size = 1000
d_model = 512
num_heads = 8
num_layers = 6
ff_dim = 2048

model = Transformer(vocab_size, d_model, num_heads, num_layers, ff_dim)
ps, st = Lux.setup(rng, model)

# Entrada de ejemplo (batch_size=32, sequence_length=50)
batch_size = 32
seq_len = 50
x = rand(rng, Float32, vocab_size, seq_len, batch_size)
y, _ = model(x, ps, st)

println("Forma de la salida: ", size(y))