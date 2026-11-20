"""
Corpus Manager for Cooking Recipe Instructions.
Milestone 1: Compile a corpus of cooking recipe instructions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from src.preprocessor import RecipeTokenizer


# Extensive seed dataset of authentic cooking recipe instructions
DEFAULT_RECIPE_STEPS = [
    # Prep & Knife Work
    "Wash and thoroughly dry all fresh produce with clean paper towels.",
    "Dice the yellow onions, celery ribs, and peeled carrots into small even pieces.",
    "Mince four cloves of fresh garlic and set aside.",
    "Chop the fresh Italian flat-leaf parsley and cilantro finely.",
    "Grate the fresh parmesan cheese using a microplane or fine box grater.",
    "Peel and finely grate fresh ginger root into a small bowl.",
    "Trim excess fat from the chicken thighs and pat completely dry with paper towels.",
    "Cut the boneless skinless chicken breasts into bite-sized one-inch cubes.",
    "Slice the red bell peppers and mushrooms into thin, uniform strips.",
    "Zest and juice one fresh lemon, discarding any seeds.",
    "Whisk the eggs, whole milk, and a pinch of salt until smooth and aerated.",
    "In a small ramekin, mix together cornstarch and cold water to form a smooth slurry.",
    "Toss the sliced apples with ground cinnamon, nutmeg, and brown sugar.",
    "Marinate the flank steak with olive oil, minced garlic, soy sauce, and lime juice for 30 minutes.",
    "Rinse the jasmine rice under cold running water until the water runs completely clear.",
    "Drain and rinse the canned chickpeas thoroughly in a colander.",
    "Soak the dried mushrooms in boiling hot water for 20 minutes until tender.",

    # Sautéing & Skillet Techniques
    "Heat two tablespoons of extra virgin olive oil in a large skillet over medium-high heat.",
    "Add the diced onions and cook until soft and translucent, about 5 to 7 minutes.",
    "Add the minced garlic and red pepper flakes, cooking for 60 seconds until fragrant.",
    "Sear the steaks in the hot cast iron skillet for 4 minutes per side for medium-rare.",
    "Melt unsalted butter in a wide frying pan over low heat until gently foaming.",
    "Sauté the sliced button mushrooms until golden brown and their liquid has evaporated.",
    "Brown the ground beef in a deep saucepan, breaking it into small crumbles with a wooden spoon.",
    "Drain the excess grease from the browned meat and discard safely.",
    "Deglaze the bottom of the skillet with dry white wine, scraping up the browned fond.",
    "Cook the bacon strips over medium heat until crispy and deep golden, about 8 minutes.",
    "Transfer the crispy bacon to a plate lined with paper towels to drain.",
    "Toss the shrimp in the hot skillet with garlic butter until pink and opaque, about 3 minutes.",
    "Stir-fry the broccoli florets and snap peas over high heat until crisp-tender.",
    "Add tomato paste to the center of the pan and caramelize for 2 minutes to deepen the flavor.",

    # Boiling, Simmering, Soups & Pasta
    "Bring a large pot of cold water to a rolling boil over high heat.",
    "Season the boiling water generously with kosher salt until it tastes like the ocean.",
    "Drop the pasta into the boiling water and stir immediately to prevent sticking.",
    "Cook the spaghetti noodles uncovered until al dente, about 8 to 10 minutes.",
    "Reserve one cup of the starchy pasta cooking water before draining.",
    "Drain the cooked pasta thoroughly in a colander in the sink.",
    "Return the drained pasta to the pot and toss with olive oil and sauce.",
    "Pour in low-sodium chicken broth and canned crushed San Marzano tomatoes.",
    "Bring the soup liquid to a gentle boil, then reduce heat to low and cover with a lid.",
    "Simmer the stew uncovered on low heat for 45 minutes until the beef is fork-tender.",
    "Skim any excess foam or oil from the surface of the simmering broth using a ladle.",
    "Whisk in heavy cream and let the sauce simmer gently until slightly thickened.",
    "Taste the soup broth and adjust the seasonings with kosher salt and black pepper.",
    "Cook the lentils in salted vegetable broth for 25 minutes until soft and creamy.",

    # Baking & Roasting
    "Preheat the conventional oven to 350 degrees F (175 degrees C).",
    "Preheat the oven to 400 degrees F and place the rack in the middle position.",
    "Line a heavy rimmed baking sheet with parchment paper or a silicone baking mat.",
    "Lightly grease a 9x13 inch baking dish with non-stick cooking spray or butter.",
    "Whisk together all-purpose flour, baking powder, baking soda, and fine salt in a mixing bowl.",
    "Cream the softened butter and granulated white sugar together until light, pale, and fluffy.",
    "Beat in the whole eggs one at a time, incorporating fully after each addition.",
    "Pour in pure vanilla extract and mix on medium speed for 30 seconds.",
    "Gently fold the dry ingredients into the wet ingredients using a rubber spatula until just combined.",
    "Fold in semisweet chocolate chips and chopped toasted pecans with a gentle hand.",
    "Pour the smooth cake batter evenly into the prepared round cake pans.",
    "Bake in the preheated oven for 25 to 30 minutes until golden brown and bubbly.",
    "Check for doneness by inserting a wooden toothpick into the center until it emerges clean.",
    "Remove the baked goods from the oven and transfer to a wire cooling rack.",
    "Allow the cake layers to cool completely in the pans for 15 minutes before inverting.",
    "Toss the potato wedges with olive oil, smoked paprika, garlic powder, and sea salt.",
    "Spread the seasoned potato wedges in a single even layer across the prepared baking sheet.",
    "Roast the mixed vegetables at 425 degrees F for 25 minutes, flipping halfway through.",
    "Broil on high for 2 to 3 minutes until the cheese topping is bubbly and golden brown.",

    # Bread & Doughs
    "In a small glass measuring cup, dissolve active dry yeast and honey in warm water.",
    "Let the yeast mixture sit undisturbed for 10 minutes until frothy and aromatic.",
    "Combine the yeast liquid with bread flour and kosher salt in the bowl of a stand mixer.",
    "Knead the bread dough on a lightly floured surface for 10 minutes until smooth and elastic.",
    "Form the kneaded dough into a tight ball and place in an oiled mixing bowl.",
    "Cover the bowl tightly with plastic wrap and let rise in a warm spot for one hour.",
    "Punch down the risen dough gently to release built-up air pockets.",
    "Divide the bread dough into equal portions and shape into tight round rolls.",
    "Score the top of the artisan sourdough loaf with a sharp razor blade or lame.",
    "Dust the shaped loaf lightly with rice flour before transferring to the hot Dutch oven.",

    # Sauces, Emulsions & Dressings
    "Melt three tablespoons of unsalted butter in a medium saucepan over medium heat.",
    "Whisk in an equal amount of all-purpose flour to form a smooth blonde roux.",
    "Cook the flour and butter roux for 2 minutes while whisking constantly to remove raw starch taste.",
    "Slowly pour in warm whole milk in a steady stream, whisking vigorously to prevent lumps.",
    "Continue to cook the white sauce until it bubbles and coats the back of a spoon.",
    "Remove the saucepan from the heat source and stir in freshly grated cheddar cheese.",
    "Whisk Dijon mustard, red wine vinegar, minced shallot, and salt in a small mixing bowl.",
    "Slowly drizzle in extra virgin olive oil while whisking continuously until emulsified.",
    "Simmer the homemade marinara sauce with fresh basil leaves for 20 minutes.",
    "Stir in cold butter cubes at the very end to give the pan sauce a glossy finish.",

    # Seasoning, Plating & Finishing Touches
    "Season generously with kosher salt and freshly cracked black pepper to taste.",
    "Squeeze fresh lemon juice over the finished dish to brighten the rich flavors.",
    "Drizzle high-quality extra virgin olive oil over the warm pasta before serving.",
    "Garnish the top with freshly chopped flat-leaf parsley and grated pecorino cheese.",
    "Sprinkle toasted sesame seeds and sliced green scallions over the noodles.",
    "Allow the roasted chicken to rest on a carving board for 10 minutes before carving.",
    "Slice the rested grilled flank steak thinly against the natural grain of the meat.",
    "Plate the creamy polenta into shallow wide pasta bowls and top with braised short ribs.",
    "Dust the chilled chocolate truffles with unsweetened cocoa powder before serving.",
    "Serve immediately while hot with crusty warm artisan bread and cold butter.",

    # Grilling & BBQ
    "Clean the outdoor grill grates with a wire brush and oil lightly with tongs and paper towel.",
    "Preheat the charcoal grill for two-zone indirect cooking over hot coals.",
    "Place the marinated chicken thighs skin-side down on the direct heat zone of the grill.",
    "Grill the burger patties for 4 minutes on the first side without pressing down.",
    "Flip the burger patties once and top each with a thick slice of sharp cheddar cheese.",
    "Brush sweet barbecue sauce over the smoked pork ribs during the final 15 minutes of grilling.",
    "Check the internal temperature with an instant-read digital meat thermometer until it reaches 165 degrees F.",

    # Vegetarian & Healthy Cooking
    "Press the block of firm tofu between paper towels with a heavy skillet to remove excess water.",
    "Cut the drained tofu into half-inch cubes and toss with cornstarch and low-sodium tamari.",
    "Steam the fresh broccoli florets in a steamer basket over boiling water for 4 minutes.",
    "Sauté fresh baby spinach leaves in olive oil until just wilted, about 2 minutes.",
    "Mash the ripe avocados in a bowl with lime juice, sea salt, diced jalapeño, and cilantro.",
    "Blend the roasted butternut squash, coconut milk, and curry powder until velvety and smooth.",
    "Toast the pine nuts in a dry nonstick pan over low heat for 3 minutes until golden and fragrant."
]


class RecipeCorpus:
    """
    Manages loading, compiling, preprocessing, and analyzing the recipe text corpus.
    """

    def __init__(self, tokenizer: Optional[RecipeTokenizer] = None):
        self.tokenizer = tokenizer or RecipeTokenizer()
        self.raw_sentences: List[str] = []
        self.tokenized_sentences: List[List[str]] = []
        self.is_compiled: bool = False

    def load_seed_data(self) -> None:
        """Loads the comprehensive default cooking recipe instructions."""
        self.raw_sentences.extend(DEFAULT_RECIPE_STEPS)

    def load_from_text_file(self, filepath: str) -> int:
        """
        Loads recipe instructions from a text file (one instruction per line or paragraph).
        Returns number of instructions loaded.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")

        loaded_count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Split lines into discrete sentences if multi-sentence
                    sentences = self.tokenizer.split_into_sentences(line)
                    for s in sentences:
                        if len(s) > 3:
                            self.raw_sentences.append(s)
                            loaded_count += 1
        return loaded_count

    def load_from_json(self, filepath: str, key: str = "instructions") -> int:
        """
        Loads recipe steps from a JSON file.
        Key can point to a list of strings or list of objects containing recipe step strings.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON corpus file not found: {filepath}")

        loaded_count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    self.raw_sentences.append(item.strip())
                    loaded_count += 1
                elif isinstance(item, dict) and key in item:
                    steps = item[key]
                    if isinstance(steps, list):
                        for step in steps:
                            if isinstance(step, str):
                                self.raw_sentences.append(step.strip())
                                loaded_count += 1
                    elif isinstance(steps, str):
                        self.raw_sentences.append(steps.strip())
                        loaded_count += 1
        return loaded_count

    def save_to_text_file(self, filepath: str) -> None:
        """Saves current compiled raw sentences to a text file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for s in self.raw_sentences:
                f.write(s + "\n")

    def compile(self, n: int = 3) -> None:
        """
        Tokenizes all raw recipe sentences, builds vocabulary, and prepares padded token lists.
        """
        if not self.raw_sentences:
            self.load_seed_data()

        # Tokenize without padding first to build vocabulary
        unpadded_tokens = [self.tokenizer.tokenize_raw(s) for s in self.raw_sentences]
        self.tokenizer.build_vocab(unpadded_tokens)

        # Pad tokens for n-gram modeling
        self.tokenized_sentences = [
            self.tokenizer.preprocess_sentence(s, n=n, replace_unk=True)
            for s in self.raw_sentences
        ]
        self.is_compiled = True

    def get_stats(self) -> Dict[str, Any]:
        """Returns key statistical metrics about the recipe corpus."""
        total_steps = len(self.raw_sentences)
        total_tokens = sum(len(tokens) for tokens in self.tokenized_sentences)
        vocab_size = len(self.tokenizer.vocab)
        avg_tokens_per_step = (
            round(total_tokens / total_steps, 2) if total_steps > 0 else 0
        )

        return {
            "total_recipe_steps": total_steps,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "avg_tokens_per_step": avg_tokens_per_step,
            "sample_steps": self.raw_sentences[:3] if self.raw_sentences else [],
        }
