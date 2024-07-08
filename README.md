from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image


model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#prompt = "create a photo of a food named Tasty Surprise whose ingredients are Olive Oil, Corn Flakes, Black Coffee, Onion, Sphagetti, Nescafe coffee, Cucumber, Brown Sugar, Healthhy Mix, Pear"
prompt="""1,Beef with Chocolate-Hazelnut Sauce and Caramelized Apples,"Beef, Hazelnuts, Dark Chocolate, Red Wine, Beef Broth, Butter, Apples, Brown Sugar, Olive Oil, Salt, Pepper","1. Season the beef with salt and pepper. 2. Heat olive oil in a large skillet over medium-high heat. 3. Sear the beef on all sides until browned, about 3-4 minutes per side for medium-rare, or to your desired doneness. 4. Remove the beef from the skillet and let it rest. 5. In the same skillet, add red wine and beef broth, scraping up any browned bits from the bottom of the pan. 6. Reduce the liquid by half over medium heat. 7. Lower the heat and stir in the chopped chocolate until melted and smooth. 8. Add the chopped hazelnuts and 1 tbsp of butter, stirring until the sauce is well combined. 9. Season with salt and pepper to taste. Set aside. 10. In another skillet, melt the remaining 1 tbsp of butter over medium heat. 11. Add the apple slices and sprinkle with brown sugar. 12. Cook, stirring occasionally, until the apples are tender and caramelized, about 8-10 minutes. 13. Slice the rested beef. 14. Arrange the beef slices on a serving platter. 15. Drizzle with the chocolate-hazelnut sauce. 16. Serve with caramelized apples on the side. 17. Garnish with additional chopped hazelnuts if desired. 18. Enjoy this unique and flavorful combination of beef, chocolate, hazelnuts, and apples."""
image = pipe(prompt).images[0]
