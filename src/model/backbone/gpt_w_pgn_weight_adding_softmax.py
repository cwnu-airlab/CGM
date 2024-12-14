import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import copy

class PgnGPT(transformers.GPT2LMHeadModel):
#class PgnGPT(transformers.LlamaForCausalLM) :    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        
        self.gate = torch.nn.Linear(config.n_embd, 1)
        self.mask = torch.nn.Linear(config.n_embd, config.vocab_size)
        #self.mask_s = torch.nn.Linear(config.

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        input_mask = kwargs.get("input_mask", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            input_mask = input_mask[:,-1:,:]

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "input_mask": input_mask, ## custom
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids = None,
        input_mask = None,
        past_key_values  = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        ): 

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        #transformer_outputs = self.model(
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
      
        # Mask 소숫점화
        past_kv = transformer_outputs.past_key_values[-1][0]
        seq_len = past_kv.shape[2]
        batch = past_kv.shape[0]
        try:
            past_kv = past_kv.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.config.n_embd) # [batch, input, embd]
        except:
            print(past_kv.shape)
            print(hidden_states.shape)
            exit() 
        past_kv = torch.softmax(self.mask(past_kv), dim=2)
        try :
            input_mask = input_mask * past_kv
        except: 
            gap = past_kv.shape[1] - input_mask.shape[1]
            temp = input_mask[:, :1, :]
            for i in range(gap) :
                input_mask = torch.cat((input_mask, temp), dim=1)
            input_mask = input_mask * past_kv

        #print('Mask: ', input_mask[0][0][3].item(), input_mask[0][0][30254].item())
        lm_logits = self.lm_head(hidden_states)
        temp_logits = torch.softmax(lm_logits, dim=2)
        #print('original: ', lm_logits[0][0][3].item(), lm_logits[0][0][30254].item())
        #print('softmax: ', temp_logits[0][0][3].item(), temp_logits[0][0][30254].item())

        # mask 곱하기 (Mask에 1이 Gate output으로 돼야 함)
        gate_value = torch.sigmoid(self.gate(hidden_states))
        temp_logits = temp_logits * gate_value
        
        local_mask = input_mask * (1-gate_value)
        temp_logits = temp_logits + local_mask
        #print("Gate/1-Gate:", gate_value[0][0], 1-gate_value[0][0])
        #print('Final: ', temp_logits[0][0][3].item(), temp_logits[0][0][30254].item())
        #input()

        loss = None

        if not return_dict:
            output = (temp_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=temp_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states, #transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
