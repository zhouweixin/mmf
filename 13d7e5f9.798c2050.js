(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{136:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return r})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return c})),a.d(t,"default",(function(){return d}));var n=a(2),i=a(9),o=(a(0),a(166)),r={id:"configuration",title:"MMF's Configuration System",sidebar_label:"Configuration System"},s={id:"notes/configuration",title:"MMF's Configuration System",description:"MMF relies on OmegaConf for its configuration system and adds some sugar on top of it. We have developed MMF as a config-first framework. Most of the parameters/settings in MMF are configurable. MMF defines some default configuration settings for its system including datasets and models. Users can then update these values via their own config or a command line dotlist.",source:"@site/docs/notes/configuration.md",permalink:"/docs/notes/configuration",editUrl:"https://github.com/facebookresearch/mmf/edit/master/website/docs/notes/configuration.md",lastUpdatedBy:"Amanpreet Singh",lastUpdatedAt:1591828867,sidebar_label:"Configuration System",sidebar:"docs",previous:{title:"Terminology and Concepts",permalink:"/docs/notes/concepts"},next:{title:"Dataset Zoo",permalink:"/docs/notes/dataset_zoo"}},c=[{value:"OmegaConf",id:"omegaconf",children:[]},{value:"Hierarchy",id:"hierarchy",children:[]},{value:"Base Defaults",id:"base-defaults",children:[]},{value:"Dataset Config",id:"dataset-config",children:[]},{value:"Model Config",id:"model-config",children:[]},{value:"User Config",id:"user-config",children:[]},{value:"Command Line Dot List Override",id:"command-line-dot-list-override",children:[]},{value:"Includes",id:"includes",children:[]},{value:"Other overrides",id:"other-overrides",children:[]},{value:"Environment Variables",id:"environment-variables",children:[]},{value:"Base Defaults Config",id:"base-defaults-config",children:[]}],l={rightToc:c};function d(e){var t=e.components,a=Object(i.a)(e,["components"]);return Object(o.b)("wrapper",Object(n.a)({},l,a,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"MMF relies on ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/"}),"OmegaConf")," for its configuration system and adds some sugar on top of it. We have developed MMF as a config-first framework. Most of the parameters/settings in MMF are configurable. MMF defines some default configuration settings for its system including datasets and models. Users can then update these values via their own config or a command line dotlist."),Object(o.b)("p",null,Object(o.b)("strong",{parentName:"p"},"TL;DR")),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"MMF uses OmegaConf for its configuration system with some sugar on top."),Object(o.b)("li",{parentName:"ul"},"MMF defines ",Object(o.b)("a",Object(n.a)({parentName:"li"},{href:"#base-defaults-config"}),"base defaults config")," containing all MMF specific parameters and then each dataset and model define their own configs (example configs: ",Object(o.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/facebookresearch/mmf/blob/master/mmf/configs/models/mmbt/defaults.yaml"}),"[model]")," ",Object(o.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/facebookresearch/mmf/blob/master/mmf/configs/datasets/hateful_memes/defaults.yaml"}),"[dataset]"),")."),Object(o.b)("li",{parentName:"ul"},"The user can define its own config specified by ",Object(o.b)("inlineCode",{parentName:"li"},"config=<x>")," at command line for each unique experiment or training setup. This has higher priority then base, model and dataset default configs and can override anything in those."),Object(o.b)("li",{parentName:"ul"},"Finally, user can override (highest priority) the final config generated by merge of all above configs by specifying config parameters as ",Object(o.b)("a",Object(n.a)({parentName:"li"},{href:"https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-dot-list"}),"dotlist")," in their command. This is the ",Object(o.b)("strong",{parentName:"li"},"recommended")," way of overriding the config parameters in MMF."),Object(o.b)("li",{parentName:"ul"},"How MMF knows which config to pick for dataset and model? The user needs to specify those in his command as ",Object(o.b)("inlineCode",{parentName:"li"},"model=x")," and ",Object(o.b)("inlineCode",{parentName:"li"},"dataset=y"),"."),Object(o.b)("li",{parentName:"ul"},"Some of the MMF config parameters under ",Object(o.b)("inlineCode",{parentName:"li"},"env")," field can be overridden by environment variable. Have a look at them.")),Object(o.b)("h2",{id:"omegaconf"},"OmegaConf"),Object(o.b)("p",null,"For understanding and using the MMF configuration system to its full extent having a look at ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/"}),"OmegaConf docs")," especially the sections on ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation"}),"interpolation"),", ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation"}),"access")," and ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/usage.html#configuration-flags"}),"configuration flags"),". MMF's config currently is by default in struct mode and we plan to make it readonly in future."),Object(o.b)("h2",{id:"hierarchy"},"Hierarchy"),Object(o.b)("p",null,"MMF follows set hierarchy rules to determine the final configuration values. Following list shows the building blocks of MMF's configuration in an increasing order of priority (higher rank will override lower rank)."),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(n.a)({parentName:"li"},{href:"#base-defaults-config"}),"Base Defaults Config")),Object(o.b)("li",{parentName:"ul"},"Dataset's Config (defined in dataset's ",Object(o.b)("inlineCode",{parentName:"li"},"config_path")," classmethod)"),Object(o.b)("li",{parentName:"ul"},"Model's Config (defined in model's ",Object(o.b)("inlineCode",{parentName:"li"},"config_path")," classmethod)"),Object(o.b)("li",{parentName:"ul"},"User's Config (Passed by user as ",Object(o.b)("inlineCode",{parentName:"li"},"config=x")," in command)"),Object(o.b)("li",{parentName:"ul"},"Command Line DotList (Passed by user as ",Object(o.b)("inlineCode",{parentName:"li"},"x.y.z=v")," dotlist in command)")),Object(o.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(o.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(o.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(o.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"Configs other than base defaults can still add new nodes that are not in base defaults config, so user can add their own config parameters if they need to without changing the base defaults. If a node has same path, nodes in higher priority config will override the lower priority nodes."))),Object(o.b)("h2",{id:"base-defaults"},"Base Defaults"),Object(o.b)("p",null,"Full base defaults config can be seen ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"#base-defaults-config"}),"below"),". This config is base of MMF's configuration system and is included in all of the experiments. It sets up nodes for training related configuration and those that need to be filled by other configs which are specified by user. Main configuration parameters that base defaults define:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"training parameters"),Object(o.b)("li",{parentName:"ul"},"distributed training parameters"),Object(o.b)("li",{parentName:"ul"},"env parameters"),Object(o.b)("li",{parentName:"ul"},"evaluation parameters"),Object(o.b)("li",{parentName:"ul"},"checkpoint parameters"),Object(o.b)("li",{parentName:"ul"},"run_type parameters")),Object(o.b)("h2",{id:"dataset-config"},"Dataset Config"),Object(o.b)("p",null,"Each dataset ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"../lib/common/registry"}),"registered")," to MMF can define its defaults config by specifying it in classmethod ",Object(o.b)("inlineCode",{parentName:"p"},"config_path")," (",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/blob/ae1689c0e2f9d8f51f337676495057168751c5ea/mmf/datasets/builders/ocrvqa/builder.py#L15"}),"example"),"). If ",Object(o.b)("inlineCode",{parentName:"p"},"processors")," key whose value is a dictionary is specified, processors will be initialized by the dataset builder. If dataset builder inherits from MMFDatasetBuilder, it will look for ",Object(o.b)("inlineCode",{parentName:"p"},"annotations"),", ",Object(o.b)("inlineCode",{parentName:"p"},"features")," and ",Object(o.b)("inlineCode",{parentName:"p"},"images")," field as well in the configuration. A sample config for a builder inheriting MMFDatasetBuilder would look like:"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"dataset_config:\n  dataset_registry_key:\n    use_images: true\n    use_features: true\n    annotations:\n      train:\n        - ...\n      val:\n        - ...\n      test:\n        - ...\n    images:\n      train:\n        - ...\n      val:\n        - ...\n      test:\n        - ...\n    features:\n      train:\n        - ...\n      val:\n        - ...\n      test:\n        - ...\n    processors:\n      text_processor:\n        type: x\n        params: ...\n")),Object(o.b)("p",null,"Configs for datasets packages with MMF are present at ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/tree/ae1689c0e2f9d8f51f337676495057168751c5ea/mmf/configs/datasets"}),"mmf/configs/datasets"),". Each dataset also provides composable configs which can be used to use some different from default but standard variation of the datasets. These can be directly included into user config by using ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"#includes"}),"includes")," directive."),Object(o.b)("p",null,"User needs to specify the dataset they are using by adding ",Object(o.b)("inlineCode",{parentName:"p"},"dataset=<dataset_key>")," option to their command."),Object(o.b)("h2",{id:"model-config"},"Model Config"),Object(o.b)("p",null,"Similar to dataset config, each model ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"../lib/common/registry"}),"registered")," to MMF can define its config. this is defined by model's ",Object(o.b)("inlineCode",{parentName:"p"},"config_path")," classmethod (",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/blob/ae1689c0e2f9d8f51f337676495057168751c5ea/mmf/models/cnn_lstm.py#L40"}),"example"),"). Configs for models live at ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/tree/ae1689c0e2f9d8f51f337676495057168751c5ea/mmf/configs/models"}),"mmf/configs/models"),". Again, like datasets models also provide some variations which can be used by including configs for those variations in the user config."),Object(o.b)("p",null,"User needs to specify the model they want to use by adding ",Object(o.b)("inlineCode",{parentName:"p"},"model=<model_key>")," option to their command. A sample model config would look like:"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"model_config:\n  model_key:\n    random_module: ...\n")),Object(o.b)("h2",{id:"user-config"},"User Config"),Object(o.b)("p",null,"User can specify their configuration specific to an experiment or training setup by adding ",Object(o.b)("inlineCode",{parentName:"p"},"config=<config_path>")," argument to their command. User config can specify for e.g. training parameters according to their experiment such as batch size using ",Object(o.b)("inlineCode",{parentName:"p"},"training.batch_size"),". Most common use case for user config is to specify optimizer, scheduler and training parameters. Other than that user config can also include configs for variations of models and datasets they want to test on. Have a look at an example user config ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/blob/master/projects/hateful_memes/configs/mmbt/defaults.yaml"}),"here"),"."),Object(o.b)("h2",{id:"command-line-dot-list-override"},"Command Line Dot List Override"),Object(o.b)("p",null,"Updating the configuration through ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-dot-list"}),"dot list")," syntax is very helpful when running multiple versions of an experiment without actually updating a config. For example, to override batch size from command line you can add ",Object(o.b)("inlineCode",{parentName:"p"},"training.batch_size=x")," at the end of your command. Similarly, for overriding an annotation in the hateful memes dataset, you can do ",Object(o.b)("inlineCode",{parentName:"p"},"dataset_config.hateful_memes.annotations.train[0]=x"),"."),Object(o.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(o.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(o.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(o.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"Command Line Dot List overrides are our recommended way of updating config parameters instead of manually updating them in config for every other change."))),Object(o.b)("h2",{id:"includes"},"Includes"),Object(o.b)("p",null,"MMF's configuration system on top of OmegaConf allows building user configs by including composable configs provided by the datasets and models. You can include it following the syntax"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"includes:\n  - path/to/first/yaml/to/be/included.yaml\n  - second.yaml\n")),Object(o.b)("p",null,"The configs will override in the sequence of how they appear in the directive. Finally, the config parameters defined in the current config will override what is present in the includes. So, for e.g."),Object(o.b)("p",null,"First file, ",Object(o.b)("inlineCode",{parentName:"p"},"a.yaml"),":"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"# a.yaml\ndataset_config:\n  hateful_memes:\n    max_features: 80\n    use_features: true\n  vqa2:\n    use_features: true\n\nmodel_config:\n  mmbt:\n    num_classes: 4\n    features_dim: 2048\n")),Object(o.b)("p",null,"Second file, ",Object(o.b)("inlineCode",{parentName:"p"},"b.yaml"),":"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"# b.yaml\noptimizer:\n  type: adam\n\ndataset_config:\n  hateful_memes:\n    max_features: 90\n    use_features: false\n    use_images: true\n  vqa2:\n    depth_first: false\n")),Object(o.b)("p",null,"And final user config, ",Object(o.b)("inlineCode",{parentName:"p"},"user.yaml"),":"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"# user.yaml\nincludes:\n  - a.yaml\n  - b.yaml\n\ndataset_config:\n  hateful_memes:\n    max_features: 100\n  vqa2:\n    annotations:\n      train: x.npy\n\nmodel_config:\n  mmbt:\n    num_classes: 2\n")),Object(o.b)("p",null,"would result in final config:"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"dataset_config:\n  hateful_memes:\n    max_features: 100\n    use_features: false\n    use_images: true\n  vqa2:\n    use_features: true\n    depth_first: false\n    annotations:\n      train: x.npy\n\nmodel_config:\n  mmbt:\n    num_classes: 2\n    features_dim: 2048\n\noptimizer:\n  type: adam\n")),Object(o.b)("h2",{id:"other-overrides"},"Other overrides"),Object(o.b)("p",null,"We also support some useful overrides schemes at the same level of command line dot list override. For example, user can specify their overrides in form of ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://pypi.org/project/demjson/"}),"demjson")," as value to argument ",Object(o.b)("inlineCode",{parentName:"p"},"--config_override")," which will them override each part of config accordingly."),Object(o.b)("h2",{id:"environment-variables"},"Environment Variables"),Object(o.b)("p",null,"MMF supports overriding some of the config parameters through environment variables. Have a look at them in ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"#base-defaults-config"}),"base default config"),"'s ",Object(o.b)("inlineCode",{parentName:"p"},"env")," parameters."),Object(o.b)("h2",{id:"base-defaults-config"},"Base Defaults Config"),Object(o.b)("p",null,"Have a look at the ",Object(o.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/mmf/blob/master/mmf/configs/defaults.yaml"}),"defaults config of MMF")," along with description of parameters from which you may need to override parameters for your experiments."))}d.isMDXComponent=!0},166:function(e,t,a){"use strict";a.d(t,"a",(function(){return m})),a.d(t,"b",(function(){return p}));var n=a(0),i=a.n(n);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function s(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function c(e,t){if(null==e)return{};var a,n,i=function(e,t){if(null==e)return{};var a,n,i={},o=Object.keys(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||(i[a]=e[a]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(i[a]=e[a])}return i}var l=i.a.createContext({}),d=function(e){var t=i.a.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):s(s({},t),e)),a},m=function(e){var t=d(e.components);return i.a.createElement(l.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return i.a.createElement(i.a.Fragment,{},t)}},f=i.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,o=e.originalType,r=e.parentName,l=c(e,["components","mdxType","originalType","parentName"]),m=d(a),f=n,p=m["".concat(r,".").concat(f)]||m[f]||b[f]||o;return a?i.a.createElement(p,s(s({ref:t},l),{},{components:a})):i.a.createElement(p,s({ref:t},l))}));function p(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=a.length,r=new Array(o);r[0]=f;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:n,r[1]=s;for(var l=2;l<o;l++)r[l]=a[l];return i.a.createElement.apply(null,r)}return i.a.createElement.apply(null,a)}f.displayName="MDXCreateElement"}}]);