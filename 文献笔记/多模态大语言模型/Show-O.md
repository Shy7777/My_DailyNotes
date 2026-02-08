在了解Show-o之前，要知道之前的多模态范式：
	LLaVA：仅理解的多模态，将输入的文本和视频通过LLM来生成文本
		benchmark,baseline
	SD3:仅生成的多模态，将输入的语言通过扩散传播生成视觉
新范式：
	Show-o，全新范式，直接把多模态融入到transformer层中，
	Show-o2，对于Show-o的改良，运用了流式